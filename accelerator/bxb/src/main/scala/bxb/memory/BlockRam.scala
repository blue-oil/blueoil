package bxb.memory

import chisel3._
import bxb.util.{Util}

// XXX: this one supposed to be replaced with black box
// verilog (direct mega function instantiation or something later)
// for now let's consider it to be a behavioral model
class BlockRam(size: Int, width: Int) extends Module {
  val addrWidth = Chisel.log2Up(size)
  val io = IO(new Bundle {
    val read = Input(ReadPort(addrWidth))
    val write = Input(WritePort(addrWidth, width))
    val readQ = Output(UInt(width.W))
  })
  val bank = SyncReadMem(size, UInt(width.W))
  io.readQ := bank.read(io.read.addr, true.B)
  when(io.write.enable) {
    bank.write(io.write.addr, io.write.data)
  }
}

object BlockRam {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new BlockRam(4096, 2)))
  }
}
