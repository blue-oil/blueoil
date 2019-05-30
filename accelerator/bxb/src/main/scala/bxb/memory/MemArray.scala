package bxb.memory

import chisel3._
import bxb.util.{Util}

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
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new MemArray(4, 4096, 2)))
  }
}
