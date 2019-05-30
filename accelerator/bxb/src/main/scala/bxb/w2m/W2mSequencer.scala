package bxb.w2m

import chisel3._

import bxb.util.{Util}

class W2mSequencer(b: Int, wAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val control = Output(W2mControl(wAddrWidth))
    // M Semaphore Dec interface
    val mWarDec = Output(Bool())
    val mWarZero = Input(Bool())
    // W Semaphore Dec interface
    val wRawDec = Output(Bool())
    val wRawZero = Input(Bool())
  })

  val syncDecMWar = RegInit(true.B)
  val syncIncMRaw = Wire(Bool())
  val syncDecWRaw = RegInit(true.B)
  val syncIncWWar = Wire(Bool())

  val waitRequired = ((syncDecMWar & io.mWarZero) | (syncDecWRaw & io.wRawZero))

  // XXX: b must be a power of two
  // we want to count from b-1 to 0
  val bCountMax = b - 1
  val bCountLeft = RegInit(bCountMax.U)
  val bCountLast = (bCountLeft === 0.U)
  when(~waitRequired) {
    when(bCountLast) {
      bCountLeft := bCountMax.U
    }.otherwise {
      bCountLeft := bCountLeft - 1.U
    }
  }

  val wAddr = RegInit(0.U(wAddrWidth.W))
  when(~waitRequired) {
    wAddr := wAddr + 1.U
  }
  
  val mWriteEnableEven = RegInit(true.B)
  val mWriteEnableOdd = RegInit(false.B)
  when(~waitRequired & bCountLast) {
    mWriteEnableEven := ~mWriteEnableEven
    mWriteEnableOdd := ~mWriteEnableOdd
  }

  io.control.wAddr := wAddr
  io.control.mWe(0) := mWriteEnableEven & ~waitRequired
  io.control.mWe(1) := mWriteEnableOdd & ~waitRequired

  // XXX: logic which dirves <MWar,MRaw> signals is same to one driving <WRaw,MWar>
  // do we need duplicate it?
  when(~waitRequired) {
    syncDecMWar := bCountLast
    syncDecWRaw := bCountLast
  }
  syncIncMRaw := bCountLast
  syncIncWWar := bCountLast

  io.mWarDec := ~waitRequired & syncDecMWar
  io.control.mRawInc := ~waitRequired & syncIncMRaw
  io.wRawDec := ~waitRequired & syncDecWRaw
  io.control.wWarInc := ~waitRequired & syncIncWWar
}
