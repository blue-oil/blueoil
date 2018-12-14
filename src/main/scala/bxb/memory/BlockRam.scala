package bxb.memory

import chisel3._

// XXX: this one supposed to be replaced with black box
// verilog (direct mega function instantiation or something later)
// for now let's consider it to be a behavioral model
class BlockRam(size: Int, width: Int) extends Module {
  val addrWidth = Chisel.log2Up(size)
  val io = IO(new Bundle {
    val readAddr = Input(UInt(addrWidth.W))
    val writeAddr = Input(UInt(addrWidth.W))
    val writeData = Input(UInt(width.W))
    val writeEnable = Input(Bool())
    val readQ = Output(UInt(width.W))
  })
  val bank = SyncReadMem(size, UInt(width.W))
  io.readQ := bank.read(io.readAddr, true.B)
  when(io.writeEnable) {
    bank.write(io.writeAddr, io.writeData)
  }
}

object BlockRam {
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new BlockRam(4096, 2)))
  }
}
