package bxb.a2f

import chisel3._

class A2fControl(private val aAddrWidth: Int, private val fAddrWidth: Int) extends Bundle {
  val accumulate = Bool()
  val writeEnable = Bool()
  val fAddr = UInt(fAddrWidth.W)
  val aAddr = UInt(aAddrWidth.W)
  val evenOdd = UInt(1.W)
  val syncInc = A2fSyncInc()
}

object A2fControl {
  def apply(aAddrWidth: Int, fAddrWidth: Int) = {
    new A2fControl(aAddrWidth, fAddrWidth)
  }
}
