package bxb.sync

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class SemaphoreTestFull(dut: Semaphore, maxValue: Int) extends PeekPokeTester(dut) {
  expect(dut.io.zero, true)
  expect(dut.io.full, false)
  poke(dut.io.decrement, false)
  for (i <- 0 until maxValue) {
    poke(dut.io.increment, true)
    step(1)
    expect(dut.io.zero, false)
    if (i != maxValue - 1) {
      expect(dut.io.full, false)
    }
    else {
      expect(dut.io.full, true)
    }
  }
}

class SemaphoreTestZero(dut: Semaphore, maxValue: Int) extends PeekPokeTester(dut) {
  expect(dut.io.zero, false)
  expect(dut.io.full, true)
  poke(dut.io.increment, false)
  for (i <- 0 until maxValue) {
    poke(dut.io.decrement, true)
    step(1)
    expect(dut.io.full, false)
    if (i != maxValue - 1) {
      expect(dut.io.zero, false)
    }
    else {
      expect(dut.io.zero, true)
    }
  }
}

class SemaphoreTestIncDec(dut: Semaphore, maxValue: Int) extends PeekPokeTester(dut) {
  expect(dut.io.zero, true)
  expect(dut.io.full, false)
  poke(dut.io.increment, true)
  poke(dut.io.decrement, false)
  step(1)
  expect(dut.io.zero, false)
  expect(dut.io.full, false)
  for (i <- 0 until 2 * maxValue) {
    poke(dut.io.increment, true)
    poke(dut.io.decrement, true)
    step(1)
    expect(dut.io.zero, false)
    expect(dut.io.full, false)
  }
}

class SemaphoreTestIncDecInZeroState(dut: Semaphore, maxValue: Int) extends PeekPokeTester(dut) {
  expect(dut.io.zero, true)
  expect(dut.io.full, false)
  poke(dut.io.increment, true)
  poke(dut.io.decrement, true)
  step(1)
  expect(dut.io.zero, false)
  expect(dut.io.full, false)
}

class SemaphoreTestIncDecInFullState(dut: Semaphore, maxValue: Int) extends PeekPokeTester(dut) {
  expect(dut.io.zero, false)
  expect(dut.io.full, true)
  poke(dut.io.increment, true)
  poke(dut.io.decrement, true)
  step(1)
  expect(dut.io.zero, false)
  expect(dut.io.full, false)
}

object SemaphoreTests {
  def main(args: Array[String]): Unit = { 
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "false", "--test-seed", "1547278051217")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    val width = 2
    val maxValue = (0x1 << width) - 1
    var ok = true
    ok &= Driver.execute(driverArgs, () => new Semaphore(width, 0))(c => new SemaphoreTestFull(c, maxValue))
    ok &= Driver.execute(driverArgs, () => new Semaphore(width, maxValue))(c => new SemaphoreTestZero(c, maxValue))
    ok &= Driver.execute(driverArgs, () => new Semaphore(width, 0))(c => new SemaphoreTestIncDec(c, maxValue))
    ok &= Driver.execute(driverArgs, () => new Semaphore(width, 0))(c => new SemaphoreTestIncDecInZeroState(c, maxValue))
    ok &= Driver.execute(driverArgs, () => new Semaphore(width, maxValue))(c => new SemaphoreTestIncDecInFullState(c, maxValue))
    if (!ok && !args.contains("noexit"))
      System.exit(1)
  }
}
