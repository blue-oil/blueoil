package bxb.a2f

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.util.{Util}

class A2fPipeline(b: Int, aAddrWidth: Int, aWidth: Int, fAddrWidth: Int, fWidth: Int) extends Module {
  val io = IO(new Bundle {
    // Pipeline signals
    val control = Input(AddrControl(aAddrWidth, fAddrWidth))
    // Systolic array interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
    val accIn = Input(Vec(b, UInt(fWidth.W)))
    // AMem interface
    val amemRead = Output(Vec(b, ReadPort(aAddrWidth)))
    val amemQ = Input(Vec(b, UInt(aWidth.W)))
    // FMem interface
    val fmemRead = Output(Vec(b, ReadPort(fAddrWidth)))
    val fmemWrite = Output(Vec(b, WritePort(fAddrWidth, fWidth)))
    val fmemQ = Input(Vec(b, UInt(fWidth.W)))
  })
  val addressPipeline = Module(new AddressPipeline(b, aAddrWidth, aWidth, fAddrWidth))
  addressPipeline.io.control := io.control
  io.amemRead := addressPipeline.io.amemRead
  addressPipeline.io.amemQ := io.amemQ
  io.aOut := addressPipeline.io.aOut
  val accumulationPipeline = Module(new AccumulationPipeline(b, fAddrWidth, fWidth))
  accumulationPipeline.io.accIn := io.accIn
  accumulationPipeline.io.accumulate := addressPipeline.io.next.accumulate
  accumulationPipeline.io.writeEnable := addressPipeline.io.next.writeEnable
  accumulationPipeline.io.address := addressPipeline.io.next.fAddr
  io.fmemRead := accumulationPipeline.io.memRead
  accumulationPipeline.io.memQ := io.fmemQ
  io.fmemWrite := accumulationPipeline.io.memWrite
}

object A2fPipeline {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new A2fPipeline(3, 10, 2, 10, 16)))
  }
}
