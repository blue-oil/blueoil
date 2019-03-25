package bxb.a2f

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class A2fSequencer(addrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val inputCCount = Input(UInt(6.W))
    val kernelVCount = Input(UInt(2.W))
    val kernelHCount = Input(UInt(2.W))
    val tileVCount = Input(UInt(addrWidth.W))
    val tileHCount = Input(UInt(addrWidth.W))
    val tileStep = Input(UInt(2.W))
    val tileGap = Input(UInt(2.W))
    val tileValid = Input(Bool())
    val tileAccepted = Output(Bool())
    val control = Output(A2fControl(addrWidth, addrWidth))
    val controlValid = Output(Bool())
    // A Semaphore Pair Dec interface
    val aRawDec = Output(Bool())
    val aRawZero = Input(Bool())
    // M Semaphore Pair Dec interface
    val mRawDec = Output(Bool())
    val mRawZero = Input(Bool())
    // F Semaphore Pair Dec interface
    val fWarDec = Output(Bool())
    val fWarZero = Input(Bool())
  })

  object State {
    val idle :: doingFirst :: doingRest :: Nil = Enum(3)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val doingFirst = (state === State.doingFirst)

  val controlWrite = RegInit(false.B)
  val controlAccumulate = RegInit(false.B)
  val controlEvenOdd = RegInit(0.U)

  // asserted at the last element of last 1x1 convolution
  val syncIncAWar = Wire(Bool())
  // asserted at the first element of first 1x1 convolution
  val syncDecARaw = RegInit(false.B)
  // asserted at the last element of 1x1 convolution
  val syncIncMWar = Wire(Bool())
  // asserted at the first element of 1x1 convolution
  val syncDecMRaw = RegInit(false.B)
  // asserted at the last element of last 1x1 convolution
  val syncIncFRaw = Wire(Bool())
  // asserted at the first element of first 1x1 convolution
  val syncDecFWar = RegInit(false.B)

  val waitRequired = ((syncDecARaw & io.aRawZero) | (syncDecMRaw & io.mRawZero) | (syncDecFWar & io.fWarZero))

  // TODO: it was preliminary optimized by inserting RegNext in front of the comparator
  // but it breaks cases of 2x2 and 1x1 output sizes, registers were removed for first two counters
  // review this one more time later
  val tileHCountLeft = Reg(UInt(addrWidth.W))
  val tileHCountLast = (tileHCountLeft === 1.U)
  when(~waitRequired) {
    when(idle | tileHCountLast) {
      tileHCountLeft := io.tileHCount
    }.otherwise {
      tileHCountLeft := tileHCountLeft - 1.U
    }
  }

  val tileVCountLeft = Reg(UInt(addrWidth.W))
  val tileVCountLast = (tileVCountLeft === 1.U) & tileHCountLast
  when(~waitRequired) {
    when(idle | tileVCountLast) {
      tileVCountLeft := io.tileVCount
    }.elsewhen(tileHCountLast) {
      tileVCountLeft := tileVCountLeft - 1.U
    }
  }

  val kernelHCountLeft = Reg(UInt(2.W))
  val kernelHCountLast = RegNext(kernelHCountLeft === 1.U) & tileVCountLast
  when(~waitRequired) {
    when(idle | kernelHCountLast) {
      kernelHCountLeft := io.kernelHCount
    }.elsewhen(tileVCountLast) {
      kernelHCountLeft := kernelHCountLeft - 1.U
    }
  }

  val kernelVCountLeft = Reg(UInt(2.W))
  val kernelVCountLast = RegNext(kernelVCountLeft === 1.U) & kernelHCountLast
  when(~waitRequired) {
    when(idle | kernelVCountLast) {
      kernelVCountLeft := io.kernelVCount
    }.elsewhen(kernelHCountLast) {
      kernelVCountLeft := kernelVCountLeft - 1.U
    }
  }

  val inputCCountLeft = Reg(UInt(6.W))
  val inputCCountLast = RegNext(inputCCountLeft === 1.U) & kernelVCountLast
  when(~waitRequired) {
    when(idle | inputCCountLast) {
      inputCCountLeft := io.inputCCount
    }.elsewhen(kernelVCountLast) {
      inputCCountLeft := inputCCountLeft - 1.U
    }
  }

  val nextIsIdle = (inputCCountLast & ~io.tileValid)

  val aAddrEvenOdd = RegInit(0.U(1.W))
  val nextAAddr = Cat(aAddrEvenOdd, 0.U((addrWidth - 1).W))
  when(~waitRequired) {
    when((idle & io.tileValid) | (kernelVCountLast & ~nextIsIdle)) {
      aAddrEvenOdd := ~aAddrEvenOdd
    }
  }

  val offset = Reg(UInt(addrWidth.W))
  when(~waitRequired) {
    when(idle | kernelVCountLast) {
      offset := nextAAddr
    }.elsewhen(kernelHCountLast) {
      offset := offset + io.tileHCount
    }.elsewhen(tileVCountLast) {
      offset := offset + 1.U
    }
  }

  val aAddr = Reg(UInt(addrWidth.W))
  when(~waitRequired) {
    when(idle | kernelVCountLast) {
      aAddr := nextAAddr
    }.elsewhen(kernelHCountLast) {
      aAddr := offset + io.tileHCount
    }.elsewhen(tileVCountLast) {
      aAddr := offset + 1.U
    }.elsewhen(tileHCountLast) {
      aAddr := aAddr + io.tileGap
    }.otherwise {
      aAddr := aAddr + io.tileStep
    }
  }

  val fAddrEvenOdd = RegInit(0.U(1.W))
  val nextFAddr = Cat(fAddrEvenOdd, 0.U((addrWidth - 1).W))
  val currenFAddr = Cat(~fAddrEvenOdd, 0.U((addrWidth - 1).W))
  when(~waitRequired) {
    when((idle & io.tileValid) | (inputCCountLast & ~nextIsIdle)) {
      fAddrEvenOdd := ~fAddrEvenOdd
    }
  }

  val fAddr = Reg(UInt(addrWidth.W))
  when(~waitRequired) {
    when(idle | inputCCountLast) {
      fAddr := nextFAddr
    }.elsewhen(tileVCountLast) {
      fAddr := currenFAddr
    }.otherwise {
      fAddr := fAddr + 1.U
    }
  }

  io.control.aAddr := aAddr
  io.control.fAddr := fAddr

  when(~waitRequired) {
    when(tileVCountLast) {
      controlEvenOdd := ~controlEvenOdd
    }
    when(idle | inputCCountLast) {
      controlAccumulate := false.B
      when(io.tileValid) {
        state := State.doingFirst
        controlWrite := true.B
      }.otherwise {
        state := State.idle
        controlWrite := false.B
      }
    }.elsewhen(doingFirst & tileVCountLast) {
      controlAccumulate := true.B
      state := State.doingRest
    }
  }

  io.control.writeEnable := ~waitRequired & controlWrite
  io.control.accumulate := controlAccumulate
  io.control.evenOdd := controlEvenOdd
  io.controlValid := controlWrite

  when(~waitRequired) {
    when(idle) {
      syncDecARaw := io.tileValid
    }.elsewhen(kernelVCountLast) {
      syncDecARaw := true.B
    }.otherwise {
      syncDecARaw := false.B
    }
  }
  syncIncAWar := kernelVCountLast

  when(~waitRequired) {
    when(idle) {
      syncDecMRaw := io.tileValid
    }.elsewhen(tileVCountLast) {
      syncDecMRaw := true.B
    }.otherwise {
      syncDecMRaw := false.B
    }
  }
  syncIncMWar := tileVCountLast

  when(~waitRequired) {
    when(idle) {
      syncDecFWar := io.tileValid
    }.elsewhen(inputCCountLast) {
      syncDecFWar := true.B
    }.otherwise {
      syncDecFWar := false.B
    }
  }
  syncIncFRaw := inputCCountLast

  io.aRawDec := ~waitRequired & syncDecARaw
  io.mRawDec := ~waitRequired & syncDecMRaw
  io.fWarDec := ~waitRequired & syncDecFWar
  io.control.syncInc.aWar := ~waitRequired & syncIncAWar
  io.control.syncInc.mWar := ~waitRequired & syncIncMWar
  io.control.syncInc.fRaw := ~waitRequired & syncIncFRaw
  io.tileAccepted := inputCCountLast
}

object A2fSequencer {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new A2fSequencer(10)))
  }
}
