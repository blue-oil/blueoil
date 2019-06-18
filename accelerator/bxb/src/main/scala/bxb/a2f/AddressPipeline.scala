package bxb.a2f

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.util.{Util}

class AddrBlock(aAddrWidth: Int, aWidth: Int, fAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Pipeline signals
    val control = Input(A2fControl(aAddrWidth, fAddrWidth))
    val next = Output(A2fControl(aAddrWidth, fAddrWidth))
    // Systolic array interface
    val aOut = Output(UInt(aWidth.W))
    val evenOddOut = Output(UInt(1.W))
    // AMem interface
    val amemRead = Output(ReadPort(aAddrWidth))
    val amemQ = Input(UInt(aWidth.W))
  })
  io.next := RegNext(io.control, 0.U.asTypeOf(io.control))
  io.aOut := io.amemQ
  // should be delayed by one cycle and passed with corresponding aOut
  io.evenOddOut := RegNext(io.control.evenOdd)
  io.amemRead.addr := io.control.aAddr
  io.amemRead.enable := true.B
}

class AddressPipeline(b: Int, aAddrWidth: Int, aWidth: Int, fAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Pipeline signals (from sequencer)
    val control = Input(A2fControl(aAddrWidth, fAddrWidth))
    // Systolic array interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
    val evenOddOut = Output(Vec(b, UInt(1.W)))
    // AMem interface
    val amemRead = Output(Vec(b, ReadPort(aAddrWidth)))
    val amemQ = Input(Vec(b, UInt(aWidth.W)))
    // Accumulation pipeline interface
    val next = Output(A2fControl(aAddrWidth, fAddrWidth))
  })
  val pipeline = Seq.fill(b){Module(new AddrBlock(aAddrWidth, aWidth, fAddrWidth))}
  for (row <- 0 until b) {
    if (row == 0) {
      pipeline(row).io.control := io.control
    }
    else {
      pipeline(row).io.control := pipeline(row - 1).io.next
    }
    io.aOut(row) := pipeline(row).io.aOut
    io.evenOddOut(row) := pipeline(row).io.evenOddOut
    io.amemRead(row) := pipeline(row).io.amemRead
    pipeline(row).io.amemQ := io.amemQ(row)
  }
  io.next := pipeline(b - 1).io.next
}

object AddressPipeline {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new AddressPipeline(3, 10, 2, 10)))
  }
}
