package bxb.a2f

import chisel3._

import bxb.memory.{ReadPort, WritePort}

class AccControl(private val addrWidth: Int) extends Bundle {
  val accumulate = Bool()
  val readAddr = UInt(addrWidth.W)
  val readEnable = Bool()
  val writeEnable = Bool()
  val writeAddr = UInt(addrWidth.W)
}

object AccControl {
  def apply(addrWidth: Int) = {
    new AccControl(addrWidth)
  }
}

class AccBlock(addrWidth: Int, accWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Pipeline signals
    val control = Input(AccControl(addrWidth))
    val next = Output(AccControl(addrWidth))
    // Systolic array interface
    val accIn = Input(UInt(accWidth.W))
    // FMem interface
    val ramRead = Output(ReadPort(addrWidth))
    val ramWrite = Output(WritePort(addrWidth, accWidth))
    val ramReadQ = Input(UInt(accWidth.W))
  })
  val pipelineRegs = RegNext(io.control, 0.U.asTypeOf(io.control))
  io.next := pipelineRegs
  io.ramRead.addr := io.control.readAddr
  io.ramRead.enable := io.control.readEnable
  io.ramWrite.addr := io.control.writeAddr
  io.ramWrite.data := Mux(io.control.accumulate, io.ramReadQ + io.accIn, io.accIn)
  io.ramWrite.enable := io.control.writeEnable
}

class AccumulationPipeline(b: Int, addrWidth: Int, accWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Systolic array iface
    val accIn = Input(Vec(b, UInt(accWidth.W)))
    // Address pipeline interface
    val accumulate = Input(Bool())
    val writeEnable = Input(Bool())
    val address = Input(UInt(addrWidth.W))
    // FMem interface
    val memRead = Output(Vec(b, ReadPort(addrWidth)))
    val memWrite = Output(Vec(b, WritePort(addrWidth, accWidth)))
    val memQ = Input(Vec(b, UInt(accWidth.W)))
    // TODO: sync iface
  })
  val accumulateDelayed = RegNext(io.accumulate, false.B)
  val writeEnableDelayed = RegNext(io.writeEnable, false.B)
  val addressDelayed = RegNext(io.address, 0.U)
  val pipeline = Seq.fill(b){Module(new AccBlock(addrWidth, accWidth))}
  for (col <- 0 until b) {
    if (col == 0) {
      pipeline(col).io.control.accumulate := accumulateDelayed
      pipeline(col).io.control.writeEnable := writeEnableDelayed
      pipeline(col).io.control.writeAddr := addressDelayed
      pipeline(col).io.control.readAddr := io.address
      pipeline(col).io.control.readEnable := io.writeEnable
    }
    else {
      pipeline(col).io.control := pipeline(col - 1).io.next
    }
    pipeline(col).io.accIn := io.accIn(col)
    io.memRead(col) := pipeline(col).io.ramRead
    pipeline(col).io.ramReadQ := io.memQ(col)
    io.memWrite(col) := pipeline(col).io.ramWrite
  }
}

object AccumulationPipeline {
  def getVerilog(dut: => chisel3.core.UserModule): String = {
    import firrtl._
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:chisel3.ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println(getVerilog(new AccumulationPipeline(3, 10, 16)))
  }
}
