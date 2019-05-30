package bxb.memory

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class PackedBlockRamSimpleReadWriteTest(c: PackedBlockRam, rowCount: Int, colCount: Int) extends PeekPokeTester(c) {
  val expected = Seq.fill(rowCount, colCount)(scala.util.Random.nextInt(2))
  poke(c.io.write.enable, true)
  for (col <- 0 until colCount) {
    poke(c.io.write.addr, col)
    for (row <- 0 until rowCount) {
      poke(c.io.write.data(row), expected(row)(col))
    }
    step(1)
  }
  poke(c.io.write.enable, false)
  for (col <- 0 until colCount) {
    poke(c.io.read.addr, col)
    step(1)
    for (row <- 0 until rowCount) {
      expect(c.io.q(row), expected(row)(col))
    }
  }
}

object PackedBlockRamTests {
  def main(args: Array[String]): Unit = {
    val ok = Driver(() => new PackedBlockRam(4, 1024, 2))(c => new PackedBlockRamSimpleReadWriteTest(c, 4, 1024))
    if (!ok && args(0) != "noexit")
      System.exit(1)
  }
}
