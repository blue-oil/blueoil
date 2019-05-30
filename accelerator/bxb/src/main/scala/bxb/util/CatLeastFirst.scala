package bxb.util

import chisel3._
import chisel3.util.{Cat}

object CatLeastFirst {
  // as native chisel Cat(seq) will make seq.first to be msb and seq.last to be lsb,
  // which is opposite to desired behavior, reverse sequence first and then cat
  def apply[T <: Bits](r: Seq[T]): UInt = {
    Cat(r.reverse)
  }
}
