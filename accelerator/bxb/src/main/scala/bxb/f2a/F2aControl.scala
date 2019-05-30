package bxb.f2a

import chisel3._

class F2aControl(private val fAddrWidth: Int, private val qAddrWidth: Int, private val aAddrWidth: Int) extends Bundle {
  val qWe = Bool()
  val syncInc = F2aSyncInc()
  val fmemAddr = UInt(fAddrWidth.W)
  val fmemReadEnable = Bool()
  val qmemAddr = UInt(qAddrWidth.W)
  val amemAddr = UInt(aAddrWidth.W)
  val amemWriteEnable = Bool()
}

object F2aControl {
  def apply(fAddrWidth: Int, qAddrWidth: Int, aAddrWidth: Int) = {
    new F2aControl(fAddrWidth, qAddrWidth, aAddrWidth)
  }
}
