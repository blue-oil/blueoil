package bxb.a2f

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class A2fSequencer(addrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val kernelVCount = Input(UInt(2.W))
    val kernelHCount = Input(UInt(2.W))
    val tileVCount = Input(UInt(addrWidth.W))
    val tileHCount = Input(UInt(addrWidth.W))
    val tileStep = Input(UInt(2.W))
    val tileGap = Input(UInt(2.W))
    val tileOffset = Input(UInt(addrWidth.W))
    val tileOffsetValid = Input(Bool())
    val control = Output(A2fControl(addrWidth, addrWidth))
    val controlValid = Output(Bool())
    val waitReq = Input(Bool())
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

  val tileHCountLeft = Reg(UInt(addrWidth.W))
  // the idea is to make combinational chains shorter
  // by feeding comparator output into delay registers
  // it will delay signal by one cycle thus last signal
  // should be generated one cycle earlier
  val tileHCountLast = RegNext(tileHCountLeft === 2.U)
  when(~io.waitReq) {
    when(idle | tileHCountLast) {
      tileHCountLeft := io.tileHCount
    }.otherwise {
      tileHCountLeft := tileHCountLeft - 1.U
    }
  }

  val tileVCountLeft = Reg(UInt(addrWidth.W))
  val tileVCountLast = RegNext(tileVCountLeft === 1.U) & tileHCountLast
  when(~io.waitReq) {
    when(idle | tileVCountLast) {
      tileVCountLeft := io.tileVCount
    }.elsewhen(tileHCountLast) {
      tileVCountLeft := tileVCountLeft - 1.U
    }
  }

  val kernelHCountLeft = Reg(UInt(2.W))
  val kernelHCountLast = RegNext(kernelHCountLeft === 1.U) & tileVCountLast
  when(~io.waitReq) {
    when(idle | kernelHCountLast) {
      kernelHCountLeft := io.kernelHCount
    }.elsewhen(tileVCountLast) {
      kernelHCountLeft := kernelHCountLeft - 1.U
    }
  }

  val kernelVCountLeft = Reg(UInt(2.W))
  val kernelVCountLast = RegNext(kernelVCountLeft === 1.U) & kernelHCountLast
  when(~io.waitReq) {
    when(idle | kernelVCountLast) {
      kernelVCountLeft := io.kernelVCount
    }.elsewhen(kernelHCountLast) {
      kernelVCountLeft := kernelVCountLeft - 1.U
    }
  }

  val offset = Reg(UInt(addrWidth.W))
  when(~io.waitReq) {
    when(idle | kernelVCountLast) {
      offset := io.tileOffset
    }.elsewhen(kernelHCountLast) {
      offset := offset + io.tileHCount
    }.elsewhen(tileVCountLast) {
      offset := offset + 1.U
    }
  }

  val aAddr = Reg(UInt(addrWidth.W))
  when(~io.waitReq) {
    when(idle | kernelVCountLast) {
      aAddr := io.tileOffset
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

  val fAddr = Reg(UInt(addrWidth.W))
  when(~io.waitReq) {
    when(idle | tileVCountLast) {
      fAddr := io.tileOffset
    }.otherwise {
      fAddr := fAddr + 1.U
    }
  }

  io.control.aAddr := aAddr
  io.control.fAddr := fAddr

  when(~io.waitReq) {
    when(tileVCountLast) {
      controlEvenOdd := ~controlEvenOdd
    }
    when(idle | kernelVCountLast) {
      controlAccumulate := false.B
      when(io.tileOffsetValid) {
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

  io.control.writeEnable := ~io.waitReq & controlWrite
  io.control.accumulate := controlAccumulate
  io.control.evenOdd := controlEvenOdd
  io.controlValid := controlWrite
}

object A2fSequencer {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new A2fSequencer(10)))
  }
}
