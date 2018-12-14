package bxb.memory

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class MemArraySimpleReadWriteTest(c: MemArray, rowCount: Int, colCount: Int) extends PeekPokeTester(c) {
  val expected = Seq.fill(rowCount, colCount)(scala.util.Random.nextInt(2))
  for (row <- 0 until rowCount) {
    poke(c.io.writeEnable(row), true)
  }
  for (col <- 0 until colCount) {
    for (row <- 0 until rowCount) {
      poke(c.io.writeAddr(row), col)
      poke(c.io.writeData(row), expected(row)(col))
    }
    step(1)
  }
  for (row <- 0 until rowCount) {
    poke(c.io.writeEnable(row), false)
  }
  for (col <- 0 until colCount) {
    for (row <- 0 until rowCount) {
      poke(c.io.readAddr(row), col)
    }
    step(1)
    for (row <- 0 until rowCount) {
      expect(c.io.readQ(row), expected(row)(col))
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
