package bxb.array

import chisel3._

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
  // XXX: it supposed to be implemented with single adder
  // we could directly instantiate altera adder megafunction in verilog to implement it
  // but for simplicity write it with just a mutex for now
  val mac = Mux(sign.toBool(), io.accIn - io.aIn, io.accIn + io.aIn)
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
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new Mac(16, 2)))
  }
}
