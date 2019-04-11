package bxb.f2a

import chisel3._
import chisel3.util._

import bxb.util.{Util}
import bxb.memory.{ReadPort}

class F2aSequencer(b: Int, fWidth: Int, qWidth: Int, aWidth: Int, fAddrWidth: Int, qAddrWidth: Int, aAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val control = Output(F2aControl())
    // Q Semaphore Pair Dec interface
    val qRawDec = Output(Bool())
    val qRawZero = Input(Bool())
    // A Semaphore Pair Dec interface
    val aWarDec = Output(Bool())
    val aWarZero = Input(Bool())
    // F Semaphore Pair Dec interface
    val fRawDec = Output(Bool())
    val fRawZero = Input(Bool())

    val writeEnable = Output(Bool())

    val hCount = Input(UInt(fAddrWidth.W))
    val wCount = Input(UInt(fAddrWidth.W))

    val fmemRead = Output(UInt(fAddrWidth.W))
    val fmemReadEnable = Output(Bool())
    val qmemRead = Output(UInt(qAddrWidth.W))
    val amemWriteAddr = Output(UInt(aAddrWidth.W))
  })
  object State {
    val idle :: doingQuantize :: doingQRead :: Nil = Enum(3)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val doingQuantize = (state === State.doingQuantize)
  val doingQRead = (state === State.doingQRead)

  val syncDecFRaw = RegInit(false.B)
  val syncIncFWar = RegInit(false.B)
  val syncIncQWar = RegInit(false.B)
  val syncDecAWar = RegInit(false.B)
  val syncIncARaw = RegInit(false.B)

  val waitRequired = (io.fRawZero | io.qRawZero | io.aWarZero)

  val wCountLeft = Reg(UInt(fAddrWidth.W))
  val wCountLast = (wCountLeft === 1.U)
  when(~waitRequired) {
    when(doingQRead | wCountLast) {
      wCountLeft := io.wCount
    }.elsewhen(doingQuantize) {
      wCountLeft := wCountLeft - 1.U
    }
  }

  val hCountLeft = Reg(UInt(fAddrWidth.W))
  val hCountLast = (hCountLeft === 1.U) & wCountLast
  when(~waitRequired) {
    when(doingQRead) {
      hCountLeft := io.hCount
    }.elsewhen(doingQuantize & wCountLast) {
      hCountLeft := hCountLeft - 1.U
    }
  }

  val aAddrEvenOdd = RegInit(0.U(1.W))
  when(~waitRequired) {
    when(doingQuantize & hCountLast) {
      aAddrEvenOdd := ~aAddrEvenOdd
    }
  }
  val aAddr = Reg(UInt(aAddrWidth.W))
  when(~waitRequired) {
    when(idle) {
      aAddr := Cat(aAddrEvenOdd, 0.U((aAddrWidth - 1).W))
    }.elsewhen(doingQuantize) {
      aAddr := aAddr + 1.U
    }
  }
  val fAddr = Reg(UInt(fAddrWidth.W))
  when(~waitRequired) {
    when(idle) {
      fAddr := Cat(aAddrEvenOdd, 0.U((fAddrWidth - 1).W))
    }.elsewhen(doingQuantize) {
      fAddr := fAddr + 1.U
    }
  }

  val qAddr = RegInit(0.U(qAddrWidth.W))
  when(~waitRequired) {
    when(doingQuantize & hCountLast) {
      qAddr := qAddr + 1.U
    }
  }

  syncIncQWar := false.B
  when(~waitRequired) {
    when(doingQRead) {
      syncIncQWar := true.B
    }
  }
  when(~waitRequired) {
    when(doingQRead) {
      syncDecFRaw := ~syncDecFRaw
    }
  }
  when(~waitRequired) {
    when(doingQuantize) {
      syncDecAWar := true.B

      when(hCountLast) {
        syncIncFWar := ~syncIncFWar
        syncIncARaw := ~syncIncARaw
      }
    }.elsewhen(doingQRead) {
      syncDecAWar := false.B
    }
  }

  when(~waitRequired) {
    when(idle) {
      state := State.doingQRead
    }.elsewhen(doingQRead) {
      state := State.doingQuantize
    }.elsewhen(doingQuantize & hCountLast) {
      state := State.idle
    }
  }


  io.writeEnable := doingQuantize

  val ffFInc = RegNext(~syncIncFWar)
  val ffAInc = RegNext(~syncIncARaw)
  io.control.syncInc.qWar := syncIncQWar
  io.control.syncInc.fWar := ~(syncIncFWar ^ ffFInc)
  io.control.syncInc.aRaw := ~(syncIncARaw ^ ffAInc)

  val ffADec = RegNext(~syncDecAWar)
  val ffFDec = RegNext(~syncDecFRaw)
  io.aWarDec := syncDecAWar && ffADec
  io.qRawDec := doingQRead
  io.fRawDec := ~(syncDecFRaw ^ ffFDec)

  io.fmemRead := fAddr
  io.fmemReadEnable := doingQuantize
  io.qmemRead := qAddr  
  io.amemWriteAddr := aAddr

  io.control.qWe := doingQRead
}

object F2aSequencer {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new F2aSequencer(10,10,10,10,10,10,10)))
  }
}
