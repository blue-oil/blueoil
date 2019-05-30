package bxb.w2m

import chisel3._

import bxb.memory.{ReadPort}
import bxb.util.{Util}

class W2mPipeline(b: Int, wAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Sequencer interface
    val control = Input(W2mControl(wAddrWidth))
    // WMem interface
    val wmemRead = Output(ReadPort(wAddrWidth))
    val wmemQ = Input(Vec(b, UInt(1.W)))
    // Mac Array interface
    val mOut = Output(Vec(b, Vec(2, UInt(1.W))))
    val mWe = Output(Vec(2, Bool()))
    // Sync interface
    val mRawInc = Output(Bool())
    val wWarInc = Output(Bool())
  })
  io.wmemRead.addr := io.control.wAddr
  io.wmemRead.enable := true.B
  // Should be delayed by one cycle to mach WMem read latency
  // XXX: broadcasting fanout may be too much...
  io.mWe := RegNext(io.control.mWe)
  io.mRawInc := RegNext(io.control.mRawInc)
  io.wWarInc := RegNext(io.control.wWarInc)
  for (row <- 0 until b) {
    for (pane <- 0 until 2) {
      io.mOut(row)(pane) := io.wmemQ(row)
    }
  }
}

object W2mPipeline {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new W2mPipeline(3, 10)))
  }
}
