package bxb.memory

import chisel3._

class MemArray(rows: Int, columns: Int, elemWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(columns)
  val io = IO(new Bundle {
    val readAddr = Input(Vec(rows, UInt(addrWidth.W)))
    val writeAddr = Input(Vec(rows, UInt(addrWidth.W)))
    val writeData = Input(Vec(rows, UInt(elemWidth.W)))
    val writeEnable = Input(Vec(rows, Bool()))
    val readQ = Output(Vec(rows, UInt(elemWidth.W)))
  })
  val banks = Seq.fill(rows)(Module(new BlockRam(columns, elemWidth)))
  for (row <- 0 until rows) {
    banks(row).io.readAddr := io.readAddr(row)
    banks(row).io.writeAddr := io.writeAddr(row)
    banks(row).io.writeData := io.writeData(row)
    banks(row).io.writeEnable := io.writeEnable(row)
    io.readQ(row) := banks(row).io.readQ
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
