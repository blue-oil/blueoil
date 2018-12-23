package bxb.test

import scala.collection._

object TestUtil {
  val mask16 = (0x1 << 16) - 1
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
    require(encodedW.length == b && encodedW(0).length == b)
    val result = mutable.Seq.fill(n, b)(0)
    for (i <- 0 until n) {
      for (j <- 0 until b) {
        var dot = 0
        for (k <- 0 until b) {
          dot = (dot + a(i)(k) * w(k)(j)) & mask16
        }
        result(i)(j) = dot
      }
    }
    result
  }

  // x: NxB dimensional matrix of 16-bit features
  // y: NxB dimensional matrix of 16-bit features
  // results are stored 16 bit unsigned values
  def matAdd(x: Seq[Seq[Int]], y: Seq[Seq[Int]]) = {
    require(x.length == y.length && x(0).length == y(0).length)
    val n = x.length
    val b = x(0).length
    val result = mutable.Seq.fill(n, b)(0)
    for (i <- 0 until n) {
      for (j <- 0 until b) {
        result(i)(j) = (x(i)(j) + y(i)(j)) & mask16
      }
    }
    result
  }
}
