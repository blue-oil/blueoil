package bxb.array

import chisel3._

class MacArray(b: Int, accWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    val aIn = Input(Vec(b, UInt(aWidth.W)))
    val mIn = Input(Vec(b, Vec(2, UInt(1.W))))
    val mWe = Input(Vec(2, Bool()))
    val evenOdd = Input(UInt(1.W))
    val accOut = Output(Vec(b, UInt(accWidth.W)))
  })
  val macs = Seq.fill(b, b){Module(new Mac(accWidth, aWidth))}
  for (row <- 0 until b) {
    for (col <- 0 until b) {
      macs(row)(col).io.aIn := (if (col == 0) io.aIn(row) else macs(row)(col - 1).io.aOut)
      macs(row)(col).io.accIn := (if (row == 0) 0.U else macs(row - 1)(col).io.accOut)
      for (pane <- 0 until 2) {
        macs(row)(col).io.mIn(pane) := (if (col == b - 1) io.mIn(row)(pane) else macs(row)(col + 1).io.mOut(pane))
        macs(row)(col).io.mWeIn(pane) := io.mWe(pane)
      }
      macs(row)(col).io.evenOdd := io.evenOdd
    }
  }
  for (col <- 0 until b) {
    io.accOut(col) := macs(b - 1)(col).io.accOut
  }
}

object MacArray {
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new MacArray(3, 16, 2)))
  }
}
