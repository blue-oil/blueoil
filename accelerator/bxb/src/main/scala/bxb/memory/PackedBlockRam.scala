package bxb.memory

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class PackedWritePort(private val addrWidth: Int, private val elemCount: Int, private val elemWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val data = Vec(elemCount, UInt(elemWidth.W))
  val enable = Bool()
}

object PackedWritePort {
  def apply(addrWidth: Int, elemCount: Int, elemWidth: Int) = {
    new PackedWritePort(addrWidth, elemCount, elemWidth)
  }
}

// Essentially it is a single block ram wrapped into interface similar to MemArray one
// Packed block ram consists of columns, and each of columns contains <rows> number of
// elements. All elements in the row are packed into a single word inside of
// underlaying block memory.
class PackedBlockRam(rows: Int, columns: Int, elemWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(columns)
  val io = IO(new Bundle {
    val write = Input(PackedWritePort(addrWidth, rows, elemWidth))
    val read = Input(ReadPort(addrWidth))
    val q = Output(Vec(rows, UInt(elemWidth.W)))
  })
  val wordWidth = rows * elemWidth
  val bank = Module(new BlockRam(columns, wordWidth))
  bank.io.read := io.read
  for (row <- 0 until rows) {
    val msb = (row + 1) * elemWidth - 1
    val lsb = row * elemWidth
    io.q(row) := bank.io.readQ(msb, lsb)
  }
  bank.io.write.addr := io.write.addr
  bank.io.write.enable := io.write.enable
  bank.io.write.data := io.write.data.reduce({ (low, high) => Cat(high, low) })
}

object PackedBlockRam {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new PackedBlockRam(4, 1024, 2)))
  }
}
