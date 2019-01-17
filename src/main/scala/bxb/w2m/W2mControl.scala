package bxb.w2m

import chisel3._

class W2mControl(private val wAddrWidth: Int) extends Bundle {
  val wAddr = UInt(wAddrWidth.W)
  val mWe = Vec(2, Bool())
}

object W2mControl {
  def apply(wAddrWidth: Int) = {
    new W2mControl(wAddrWidth)
  }
}
