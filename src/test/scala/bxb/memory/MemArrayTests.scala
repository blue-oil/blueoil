package bxb.memory

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class MemArraySimpleReadWriteTest(c: MemArray, rowCount: Int, colCount: Int) extends PeekPokeTester(c) {
  val expected = Seq.fill(rowCount, colCount)(scala.util.Random.nextInt(2))
  for (row <- 0 until rowCount) {
    poke(c.io.write(row).enable, true)
  }
  for (col <- 0 until colCount) {
    for (row <- 0 until rowCount) {
      poke(c.io.write(row).addr, col)
      poke(c.io.write(row).data, expected(row)(col))
    }
    step(1)
  }
  for (row <- 0 until rowCount) {
    poke(c.io.write(row).enable, false)
  }
  for (col <- 0 until colCount) {
    for (row <- 0 until rowCount) {
      poke(c.io.read(row).addr, col)
    }
    step(1)
    for (row <- 0 until rowCount) {
      expect(c.io.q(row), expected(row)(col))
    }
  }
}

object MemArrayTests {
  def main(args: Array[String]): Unit = {
    val ok = Driver(() => new MemArray(4, 4096, 2))(c => new MemArraySimpleReadWriteTest(c, 4, 4096))
    if (!ok && args(0) != "noexit")
      System.exit(1)
  }
}
