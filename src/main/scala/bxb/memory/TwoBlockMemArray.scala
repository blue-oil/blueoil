package bxb.memory

import chisel3._

class TwoBlockMemArray(rows: Int, columns: Int, elemWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(columns)
  val io = IO(new Bundle {
    val readA = Input(Vec(rows, ReadPort(addrWidth)))
    val writeA = Input(Vec(rows, WritePort(addrWidth, elemWidth)))
    val qA = Output(Vec(rows, UInt(elemWidth.W)))
    val readB = Input(Vec(rows, ReadPort(addrWidth)))
    val qB = Output(Vec(rows, UInt(elemWidth.W)))
  })
  val banks = Seq.fill(rows){Module(new TwoBlockRam(columns, elemWidth))}
  for (row <- 0 until rows) {
    banks(row).io.readA := io.readA(row)
    banks(row).io.writeA := io.writeA(row)
    io.qA(row) := banks(row).io.qA
    banks(row).io.readB := io.readB(row)
    io.qB(row) := banks(row).io.qB
  }
}

object TwoBlockMemArray {
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new TwoBlockMemArray(3, 4096 * 2, 2)))
  }
}
