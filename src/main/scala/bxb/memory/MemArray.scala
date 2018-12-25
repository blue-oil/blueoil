package bxb.memory

import chisel3._

class MemArray(rows: Int, columns: Int, elemWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(columns)
  val io = IO(new Bundle {
    val read = Input(Vec(rows, ReadPort(addrWidth)))
    val write = Input(Vec(rows, WritePort(addrWidth, elemWidth)))
    val q = Output(Vec(rows, UInt(elemWidth.W)))
  })
  val banks = Seq.fill(rows)(Module(new BlockRam(columns, elemWidth)))
  for (row <- 0 until rows) {
    banks(row).io.read := io.read(row)
    banks(row).io.write := io.write(row)
    io.q(row) := banks(row).io.readQ
  }
}


object MemArray {
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new MemArray(4, 4096, 2)))
  }
}
