package bxb.avalon

import chisel3._

class ReadMasterIO(private val addrWidth: Int, private val dataWidth: Int) extends Bundle {
  val address = Output(UInt(addrWidth.W))
  val read = Output(Bool())
  val burstCount = Output(UInt(10.W)) // FIXME: select size based on maxBurst
  val waitRequest = Input(Bool())
  val readDataValid = Input(Bool())
  val readData = Input(UInt(dataWidth.W))
}

object ReadMasterIO {
  def apply(addrWidth: Int, dataWidth: Int) = {
    new ReadMasterIO(addrWidth, dataWidth)
  }
}
