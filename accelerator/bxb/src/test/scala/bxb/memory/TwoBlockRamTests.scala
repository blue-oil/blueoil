package bxb.memory

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class TwoBlockRamTests(dut: TwoBlockRam, size: Int, dataWidth: Int) extends PeekPokeTester(dut) {
  val expected = Seq.fill(size)(scala.util.Random.nextInt(0x1 << dataWidth))
  val half = size / 2
  poke(dut.io.readB.enable, false)
  for (row <- 0 until half) {
    poke(dut.io.writeA.addr, row)
    poke(dut.io.writeA.data, expected(row))
    poke(dut.io.writeA.enable, true)
    if (row > 0) {
      poke(dut.io.readA.addr, row - 1)
      poke(dut.io.readA.enable, true)
    }
    step(1)
    if (row > 0) {
      expect(dut.io.qA, expected(row - 1))
    }
  }
  poke(dut.io.writeA.enable, false)
  poke(dut.io.readA.addr, half - 1)
  poke(dut.io.readA.enable, true)
  step(1)
  poke(dut.io.readA.enable, false)
  expect(dut.io.qA, expected(half - 1))
  for (row <- 0 until half) {
    poke(dut.io.writeA.addr, half + row)
    poke(dut.io.writeA.data, expected(half + row))
    poke(dut.io.writeA.enable, true)
    if (row > 0) {
      poke(dut.io.readA.addr, half + row - 1)
      poke(dut.io.readA.enable, true)
    }
    poke(dut.io.readB.addr, row)
    poke(dut.io.readB.enable, true)
    step(1)
    if (row > 0) {
      expect(dut.io.qA, expected(half + row - 1))
    }
    expect(dut.io.qB, expected(row))
  }
}

object TwoBlockRamTests {
  def main(args: Array[String]): Unit = {
    val ok = Driver(() => new TwoBlockRam(4096, 16))(c => new TwoBlockRamTests(c, 4096, 16))
    if (!ok && args(0) != "noexit")
      System.exit(1)
  }
}
