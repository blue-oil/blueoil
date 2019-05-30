package bxb.array

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

import bxb.test.{TestUtil}

class MacArrayTests(c: MacArray, b: Int) extends PeekPokeTester(c) {
  def fillM(pane: Int, m: Seq[Seq[Int]]) = {
    poke(c.io.mWe(pane), true)
    for (col <- 0 until b) {
      for (row <- 0 until b) {
        poke(c.io.mIn(row)(pane), m(row)(col));
      }
      step(1)
    }
    poke(c.io.mWe(pane), false)
  }

  // Actual Test Bench Logic
  var pane = 0;
  for (n <- List(b, 2 * b, 3 * b, 8 * b)) {
    // Test Data
    // weight matrix BxB
    val weights = Seq.fill(b, b)(scala.util.Random.nextInt(2))
    // input matrix is of size NxB
    val inputs = Seq.fill(n, b)(scala.util.Random.nextInt(4))
    val outputs = TestUtil.matMul(inputs, weights)

    // Fill M with test inputs
    fillM(pane, weights)
    for (col <- 0 until b) {
      poke(c.io.evenOddIn(col), pane)
    }

    // Run & Compare
    val colDelay = 0 until b
    val resDelay = b until (2 * b)
    for (t <- 0 until (n + 2 * b)) {
      for (col <- 0 until b) {
        if (t >= colDelay(col) && t < colDelay(col) + n) {
          poke(c.io.aIn(col), inputs(t - colDelay(col))(col))
        }
        if (t >= resDelay(col) && t < resDelay(col) + n) {
          expect(c.io.accOut(col), outputs(t - resDelay(col))(col))
        }
      }
      step(1)
    }
    pane = pane ^ 1;
  }
}

object MacArrayTests {
  def main(args: Array[String]): Unit = { 
    for (b <- List(2, 3, 4, 8, 16)) {
      println(f"Tesing with B = $b")
      val ok = Driver(() => new MacArray(b, 16, 2))(c => new MacArrayTests(c, b))
      if (!ok && args(0) != "noexit")
        System.exit(1)
    }
  }
}
