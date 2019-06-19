package bxb.avalon

import chisel3._

class WriteMasterIO(private val avalonAddrWidth: Int, private val avalonDataWidth: Int) extends Bundle {
    val address = Output(UInt(avalonAddrWidth.W))
    val burstCount = Output(UInt(10.W))
    val waitRequest = Input(Bool())
    val write = Output(Bool())
    val writeData = Output(UInt(avalonDataWidth.W))
}

object WriteMasterIO {
  def apply(addrWidth: Int, dataWidth: Int) = {
    new WriteMasterIO(addrWidth, dataWidth)
  }
}
