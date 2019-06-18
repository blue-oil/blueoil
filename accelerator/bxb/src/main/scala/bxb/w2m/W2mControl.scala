package bxb.w2m

import chisel3._

class W2mControl(private val wAddrWidth: Int) extends Bundle {
  // WMem control
  val wAddr = UInt(wAddrWidth.W)
  // M control
  val mWe = Vec(2, Bool())
  // Sync control
  val mRawInc = Bool()
  val wWarInc = Bool()
}

object W2mControl {
  def apply(wAddrWidth: Int) = {
    new W2mControl(wAddrWidth)
  }
}
