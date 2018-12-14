package bxb.array

import chisel3._
import scala.collection._
import chisel3.iotesters.{PeekPokeTester, Driver}

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

  // a: NxB dimensional matrix of two bit activations
  // w: BxB matrix of 1 bit weights
  //  each weight is encoded
  //   0: 1
  //   1: -1
  // results are accumulated as 16 bit unsigned values
  def matMul(a: Seq[Seq[Int]], encodedW: Seq[Seq[Int]]) = {
    val w = encodedW.map({ row => row.map({ x => if (x == 0) 1 else -1 })})
    val n = a.length
    val b = a(0).length
    val mask = (0x1 << 16) - 1
    require(encodedW.length == b && encodedW(0).length == b)
    val result = mutable.Seq.fill(n, b)(0)
    for (i <- 0 until n) {
      for (j <- 0 until b) {
        var dot = 0
        for (k <- 0 until b) {
            dot = (dot + a(i)(k) * w(k)(j)) & mask;
        }
        result(i)(j) = dot
      }
    }
    result
  }

  // Actual Test Bench Logic
  var pane = 0;
  for (n <- List(b, 2 * b, 3 * b, 8 * b)) {
    // Test Data
    // weight matrix BxB
    val weights = Seq.fill(b, b)(scala.util.Random.nextInt(2))
    // input matrix is of size NxB
    val inputs = Seq.fill(n, b)(scala.util.Random.nextInt(4))
    val outputs = matMul(inputs, weights)

    // Fill M with test inputs
    fillM(pane, weights)
    poke(c.io.evenOdd, pane)

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
