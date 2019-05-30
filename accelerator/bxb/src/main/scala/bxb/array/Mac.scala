package bxb.array

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class Mac(accWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    val aIn = Input(UInt(aWidth.W))
    val accIn = Input(UInt(accWidth.W))
    val evenOddIn = Input(UInt(1.W))
    val mIn = Input(Vec(2, UInt(1.W)))
    val mWeIn = Input(Vec(2, Bool()))
    val aOut = Output(UInt(aWidth.W))
    val accOut = Output(UInt(accWidth.W))
    val mOut = Output(Vec(2, UInt(1.W)))
    val evenOddOut = Output(UInt(1.W))
  })
  val m = Seq.fill(2)(Reg(UInt(1.W)))
  val sign = Mux(io.evenOddIn.toBool(), m(1), m(0))
  val aZeroExtend = Cat(0.U((accWidth - aWidth).W), io.aIn)
  // XXX: expected to become adder with carry in
  val mac = io.accIn + Mux(sign.toBool(), ~aZeroExtend, aZeroExtend) + sign
  val aNext = RegNext(io.aIn)
  val accNext = RegNext(mac)
  val evenOddNext = RegNext(io.evenOddIn)
  io.aOut := aNext
  io.accOut := accNext
  io.evenOddOut := evenOddNext
  for (i <- 0 until 2) {
    when(io.mWeIn(i)) {
      m(i) := io.mIn(i)
    }
    io.mOut(i) := m(i)
  }
}

object Mac {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new Mac(16, 2)))
  }
}
