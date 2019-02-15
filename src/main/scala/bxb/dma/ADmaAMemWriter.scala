package bxb.dma

import chisel3._
import chisel3.util._

import bxb.memory.{WritePort}
import bxb.util.{Util}

class ADmaAMemWriter(b: Int, avalonDataWidth: Int, aAddrWidth: Int, tileCountWidth: Int) extends Module {
  val aSz = 2
  // current data format assumes 32 activations packed into two 32 bit words:
  // first contains 32 lsb bits, second contains 32 msb bits
  val aBitPackSize = 32

  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(b >= aBitPackSize && avalonDataWidth >= 2 * aBitPackSize,
    "to simplify logic we assume that we could receive amount of data just right to write b elements in one cycle")

  val io = IO(new Bundle {
    // Tile Generator interface
    val tileHeight = Input(UInt(aAddrWidth.W))
    val tileWidth = Input(UInt(aAddrWidth.W))
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
    val idle :: running :: acknowledge :: Nil = Enum(3)
  }
  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val running = (state === State.running)
  val acknowledge = (state === State.acknowledge)

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

  // pointer to next half of AMem to be used
  val amemBufEvenOdd = RegInit(0.U(1.W))
  when(idle & io.tileValid) {
    amemBufEvenOdd := ~amemBufEvenOdd
  }

  val amemAddress = Reg(UInt(aAddrWidth.W))
  when(~waitRequired) {
    when(idle) {
      amemAddress := Cat(amemBufEvenOdd, 0.U((aAddrWidth - 1).W))
    }.otherwise {
      amemAddress := amemAddress + 1.U
    }
  }

  when(idle & io.tileValid) {
    state := State.running
  }.elsewhen(running & tileYCountLast & ~waitRequired) {
    state := State.acknowledge
  }.elsewhen(acknowledge) {
    state := State.idle
  }

  io.tileAccepted := acknowledge
  io.writerDone := acknowledge

  for (row <- 0 until b) {
    // I wish we switch to less hacky representation
    // before we start support higher bit activations
    require(aSz == 2, "below assumes aSz to be 2 bits for now")

    val lsbWord = row / aBitPackSize * 2 // 0, 2, 4 ...
    val lsbPos = lsbWord * aBitPackSize + row % aBitPackSize

    val msbWord = row / aBitPackSize * 2 + 1 // 1, 3, 5
    val msbPos = msbWord * aBitPackSize + row % aBitPackSize

    io.amemWrite(row).addr := amemAddress
    io.amemWrite(row).data := Cat(io.avalonMasterReadData(msbPos), io.avalonMasterReadData(lsbPos))
    io.amemWrite(row).enable := (running & io.avalonMasterReadDataValid)
  }
}

object ADmaAMemWriter {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new ADmaAMemWriter(32, 64, 12, 12)))
  }
}
