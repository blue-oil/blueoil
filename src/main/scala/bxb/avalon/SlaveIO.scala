package bxb.avalon

import chisel3._

class SlaveIO(private val addrWidth: Int, private val dataWidth: Int) extends Bundle {
  val address = Input(UInt(addrWidth.W))
  val writeData = Input(UInt(dataWidth.W))
  val write = Input(Bool())
  val read = Input(Bool())
  val readData = Output(UInt(dataWidth.W))
}

object SlaveIO {
  def apply(addrWidth: Int, dataWidth: Int) = {
    new SlaveIO(addrWidth, dataWidth)
  }
}
