package bxb.array

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class MacTestMLoading(c: Mac, pane: Int) extends PeekPokeTester(c) {
  poke(c.io.mIn(pane), 0)
  poke(c.io.mWeIn(pane), true)
  step(1)
  poke(c.io.mWeIn(pane), false)
  expect(c.io.mOut(pane), 0)
  step(1)
  poke(c.io.mWeIn(pane), false)
  expect(c.io.mOut(pane), 0)
  step(1)
  poke(c.io.mWeIn(pane), true)
  poke(c.io.mIn(pane), 1)
  expect(c.io.mOut(pane), 0)
  step(1)
  expect(c.io.mOut(pane), 1)
}

class MacTestAPropagation(c: Mac) extends PeekPokeTester(c) {
  poke(c.io.aIn, 0)
  step(1)
  poke(c.io.aIn, 1)
  expect(c.io.aOut, 0)
  step(1)
  poke(c.io.aIn, 0)
  expect(c.io.aOut, 1)
}

class MacTestAccumulation(c: Mac) extends PeekPokeTester(c) {
  def check(acc: Int, a: Int, expected: Int) = {
    poke(c.io.accIn, acc)
    poke(c.io.aIn, a)
    step(1)
    expect(c.io.accOut, expected)
  }

  def setupM(pane: Int, m: Int) = {
    poke(c.io.mIn(pane), m)
    poke(c.io.mWeIn(pane), true)
    step(1)
    poke(c.io.mWeIn(pane), false)
  }

  // setup m[0] to +1
  setupM(0, 0)
  poke(c.io.evenOddIn, 0)

  check(123, 1, 124)
  check(123, 2, 125)
  check(123, 0, 123)
  check(123, 3, 126)

  // setup m[1] to -1
  setupM(1, 1)
  poke(c.io.evenOddIn, 1)
  check(123, 1, 122)
  check(123, 2, 121)
  check(123, 0, 123)
  check(123, 3, 120)
}

object MacTests {
  def main(args: Array[String]): Unit = { 
    def check(ok: Boolean) = {
      if (!ok && args(0) != "noexit")
        System.exit(1)
    }
    for (pane <- 0 until 2) {
      check(Driver(() => new Mac(16, 2))(c => new MacTestMLoading(c, pane)))
    }
    check(Driver(() => new Mac(16, 2))(c => new MacTestAPropagation(c)))
    check(Driver(() => new Mac(16, 2))(c => new MacTestAccumulation(c)))
  }
}
