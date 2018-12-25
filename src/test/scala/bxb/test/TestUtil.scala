package bxb.test

import scala.collection._

object TestUtil {
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
}
