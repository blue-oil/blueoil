package bxb.memory

import chisel3._

import bxb.util.{Util}

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
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new TwoBlockMemArray(3, 4096 * 2, 2)))
  }
}
