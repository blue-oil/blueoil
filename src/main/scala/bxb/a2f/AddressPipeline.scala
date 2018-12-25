package bxb.a2f

import chisel3._

import bxb.memory.{ReadPort, WritePort}

class AddrControl(private val aAddrWidth: Int, private val fAddrWidth: Int) extends Bundle {
  val accumulate = Bool()
  val writeEnable = Bool()
  val fAddr = UInt(fAddrWidth.W)
  val aAddr = UInt(aAddrWidth.W)
  // TODO: sync
}

object AddrControl {
  def apply(aAddrWidth: Int, fAddrWidth: Int) = {
    new AddrControl(aAddrWidth, fAddrWidth)
  }
}

class AddrBlock(aAddrWidth: Int, aWidth: Int, fAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Pipeline signals
    val control = Input(AddrControl(aAddrWidth, fAddrWidth))
    val next = Output(AddrControl(aAddrWidth, fAddrWidth))
    // Systolic array interface
    val aOut = Output(UInt(aWidth.W))
    // AMem interface
    val amemRead = Output(ReadPort(aAddrWidth))
    val amemQ = Input(UInt(aWidth.W))
  })
  io.next := RegNext(io.control, 0.U.asTypeOf(io.control))
  io.aOut := io.amemQ
  io.amemRead.addr := io.control.aAddr
  io.amemRead.enable := true.B
}

class AddressPipeline(b: Int, aAddrWidth: Int, aWidth: Int, fAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Pipeline signals (from sequencer)
    val control = Input(AddrControl(aAddrWidth, fAddrWidth))
    // Systolic array interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
    // AMem interface
    val amemRead = Output(Vec(b, ReadPort(aAddrWidth)))
    val amemQ = Input(Vec(b, UInt(aWidth.W)))
    // Accumulation pipeline interface
    val next = Output(AddrControl(aAddrWidth, fAddrWidth))
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
    io.amemRead(row) := pipeline(row).io.amemRead
    pipeline(row).io.amemQ := io.amemQ(row)
  }
  io.next := pipeline(b - 1).io.next
}

object AddressPipeline {
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new AddressPipeline(3, 10, 2, 10)))
  }
}
