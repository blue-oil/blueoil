package bxb.adma

import chisel3._
import chisel3.util._

import bxb.memory.{WritePort}
import bxb.util.{Util}

class ADmaAMemWriter(b: Int, avalonDataWidth: Int, aAddrWidth: Int, tileCountWidth: Int) extends Module {
  val aSz = 2
  // current data format assumes 32 activations packed into two 32 bit words:
  // first contains 32 lsb bits, second contains 32 msb bits
  val aBitPackSize = 32
  val aAddrLowWidth = aAddrWidth - 1

  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(b >= aBitPackSize && avalonDataWidth >= 2 * aBitPackSize,
    "to simplify logic we assume that we could receive amount of data just right to write b elements in one cycle")

  val io = IO(new Bundle {
    // Tile Generator interface
    val tileHeight = Input(UInt(aAddrWidth.W))
    val tileWidth = Input(UInt(aAddrWidth.W))
    val tileStartPad = Input(UInt(aAddrWidth.W))
    val tileSidePad = Input(UInt(aAddrWidth.W))
    val tileEndPad = Input(UInt(aAddrWidth.W))
    val tileFirst = Input(Bool())
    // once tileValid asserted above tile parameters must remain stable and until tileAccepted is asserted
    val tileValid = Input(Bool())
    // accepted at a clock when last request is sent
    val tileAccepted = Output(Bool())

    val avalonMasterReadDataValid = Input(Bool())
    val avalonMasterReadData = Input(UInt(avalonDataWidth.W))

    // Avalon Requester interface
    // avalon expected to complete before writer
    val writerDone = Output(Bool())

    // AMem interface
    val amemWrite = Output(Vec(b, WritePort(aAddrWidth, aSz)))
  })
  // Destination Address Generator
  object State {
    val idle :: running :: acknowledge :: padStart :: padMiddle :: padEnd :: Nil = Enum(6)
  }
  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val running = (state === State.running)
  val acknowledge = (state === State.acknowledge)
  val padStart = (state === State.padStart)
  val padMiddle = (state === State.padMiddle)
  val padEnd = (state === State.padEnd)

  val waitRequired = (running & ~io.avalonMasterReadDataValid)

  // tile loops
  val tileXCountLeft = Reg(UInt(tileCountWidth.W))
  val tileXCountLast = (tileXCountLeft === 1.U)
  when(~waitRequired) {
    when(idle | tileXCountLast) {
      tileXCountLeft := io.tileWidth
    }.otherwise {
      tileXCountLeft := tileXCountLeft - 1.U
    }
  }

  val tileYCountLeft = Reg(UInt(tileCountWidth.W))
  val tileYCountLast = (tileYCountLeft === 1.U) & tileXCountLast
  when(~waitRequired) {
    when(idle | tileYCountLast) {
      tileYCountLeft := io.tileHeight
    }.elsewhen(tileXCountLast) {
      tileYCountLeft := tileYCountLeft - 1.U
    }
  }

  // padding loops
  val padStartCountLeft = Reg(UInt(tileCountWidth.W))
  val padStartCountLast = (padStartCountLeft === 1.U)
  when(idle) {
    padStartCountLeft := io.tileStartPad
  }.elsewhen(padStart) {
    padStartCountLeft := padStartCountLeft - 1.U
  }

  val padSideCountLeft = Reg(UInt(tileCountWidth.W))
  val padSideCountLast = (padSideCountLeft === 1.U)
  when(idle | padSideCountLast) {
    padSideCountLeft := io.tileSidePad
  }.elsewhen(padMiddle) {
    padSideCountLeft := padSideCountLeft - 1.U
  }

  val padMiddleCountLeft = Reg(UInt(tileCountWidth.W))
  val padMiddleCountLast = (padMiddleCountLeft === 1.U) & padSideCountLast
  when(idle) {
    padMiddleCountLeft := io.tileHeight - 1.U
  }.elsewhen(padMiddle & padSideCountLast) {
    padMiddleCountLeft := padMiddleCountLeft - 1.U
  }

  val padEndCountLeft = Reg(UInt(tileCountWidth.W))
  val padEndCountLast = (padEndCountLeft === 1.U)
  when(idle) {
    padEndCountLeft := io.tileEndPad
  }.elsewhen(padEnd) {
    padEndCountLeft := padEndCountLeft - 1.U
  }

  val amemAddressMsb = RegInit(0.U(1.W))
  when(idle & io.tileValid & io.tileFirst) {
    amemAddressMsb := 0.U
  }.elsewhen(idle & io.tileValid) {
    amemAddressMsb := ~amemAddressMsb
  }

  val hasStartPad = RegNext(io.tileStartPad =/= 0.U)
  val hasSidePad = RegNext(io.tileSidePad =/= 0.U)
  val hasEndPad = RegNext(io.tileEndPad =/= 0.U)

  val amemAddressLow = Reg(UInt(aAddrLowWidth.W))
  when(~waitRequired) {
    when(idle) {
      amemAddressLow := io.tileStartPad
    }.elsewhen(running) {
      // when all data from memory is written move amemAddressLow to start write padding
      when(tileYCountLast) {
        when(hasStartPad) {
          amemAddressLow := 0.U
        }.elsewhen(hasSidePad) {
          amemAddressLow := io.tileWidth
        }.otherwise {
          amemAddressLow := amemAddressLow + 1.U
        }
      }.elsewhen(tileXCountLast) {
        amemAddressLow := amemAddressLow + io.tileSidePad + 1.U
      }.otherwise {
        amemAddressLow := amemAddressLow + 1.U
      }
    }.elsewhen(padStart) {
      when(padStartCountLast) {
        amemAddressLow := amemAddressLow + io.tileWidth + 1.U
      }.otherwise {
        amemAddressLow := amemAddressLow + 1.U
      }
    }.elsewhen(padMiddle) {
      when(padSideCountLast) {
        amemAddressLow := amemAddressLow + io.tileWidth + 1.U
      }.otherwise {
        amemAddressLow := amemAddressLow + 1.U
      }
    }.elsewhen(padEnd) {
      amemAddressLow := amemAddressLow + 1.U
    }
  }

  when(idle & io.tileValid) {
    state := State.running
  }.elsewhen(running & tileYCountLast & ~waitRequired) {
    when(hasStartPad) {
      state := State.padStart
    }.elsewhen(hasSidePad) {
      state := State.padMiddle
    }.elsewhen(hasEndPad) {
      state := State.padEnd
    }.otherwise {
      state := State.acknowledge
    }
  }.elsewhen(padStart & padStartCountLast) {
    when(hasSidePad) {
      state := State.padMiddle
    }.elsewhen(hasEndPad) {
      state := State.padEnd
    }.otherwise {
      state := State.acknowledge
    }
  }.elsewhen(padMiddle & padMiddleCountLast) {
    when(hasEndPad) {
      state := State.padEnd
    }.otherwise {
      state := State.acknowledge
    }
  }.elsewhen(padEnd & padEndCountLast) {
    state := State.acknowledge
  }.elsewhen(acknowledge) {
    state := State.idle
  }

  io.tileAccepted := acknowledge
  io.writerDone := acknowledge

  val writeEnable = ((running & io.avalonMasterReadDataValid) | padStart | padMiddle | padEnd)
  for (row <- 0 until b) {
    // I wish we switch to less hacky representation
    // before we start support higher bit activations
    require(aSz == 2, "below assumes aSz to be 2 bits for now")

    val lsbWord = row / aBitPackSize * 2 // 0, 2, 4 ...
    val lsbPos = lsbWord * aBitPackSize + row % aBitPackSize

    val msbWord = row / aBitPackSize * 2 + 1 // 1, 3, 5
    val msbPos = msbWord * aBitPackSize + row % aBitPackSize

    io.amemWrite(row).addr := Cat(amemAddressMsb, amemAddressLow)
    io.amemWrite(row).data := Mux(running, Cat(io.avalonMasterReadData(msbPos), io.avalonMasterReadData(lsbPos)), 0.U(aSz.W))
    io.amemWrite(row).enable := writeEnable
  }
}

object ADmaAMemWriter {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new ADmaAMemWriter(32, 64, 12, 12)))
  }
}
