package bxb.memory

import chisel3._

class ReadPort(private val addrWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val enable = Bool()
}

object ReadPort {
  def apply(addrWidth: Int) = {
    new ReadPort(addrWidth)
  }
}

class WritePort(private val addrWidth: Int, private val dataWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val data = UInt(dataWidth.W)
  val enable = Bool()
}

object WritePort {
  def apply(addrWidth: Int, dataWidth: Int) = {
    new WritePort(addrWidth, dataWidth)
  }
}
