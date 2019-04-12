package bxb.f2a

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.util.{Util}

class F2aPipeline(b: Int, fWidth: Int, qWidth: Int, aWidth: Int, fAddrWidth: Int, qAddrWidth: Int, aAddrWidth: Int) extends Module {
  val io = IO(new Bundle{
    val control = Input(F2aControl(fAddrWidth, qAddrWidth, aAddrWidth))

    val fmemRead = Output(Vec(b, ReadPort(fAddrWidth)))
    val fmemQ = Input(Vec(b, UInt(fWidth.W)))

    val qmemRead = Output(ReadPort(qAddrWidth))
    val qmemQ = Input(Vec(b, UInt(40.W)))

    val amemWrite = Output(Vec(b, WritePort(aAddrWidth, aWidth)))
    val syncInc = Output(F2aSyncInc())
  })
  val quantizer = Module(new QuantizePipeline(b, fWidth, aWidth))
  val amemAddrBuf = RegNext(io.control.amemAddr)
  val amemWriteEnableBuf = RegNext(io.control.amemWriteEnable, init=false.B)

  quantizer.io.qWe := io.control.qWe

  for (col <- 0 until b) {
    io.fmemRead(col).addr := io.control.fmemAddr
    io.fmemRead(col).enable := io.control.fmemReadEnable
  }
  quantizer.io.fMemQ := io.fmemQ

  io.qmemRead.addr := io.control.qmemAddr
  io.qmemRead.enable := true.B
  quantizer.io.qMemQ := io.qmemQ

  for (col <- 0 until b) {
    io.amemWrite(col).data := quantizer.io.aOut(col)
    io.amemWrite(col).addr := amemAddrBuf
    io.amemWrite(col).enable := amemWriteEnableBuf
  }

  io.syncInc := RegNext(io.control.syncInc, init=0.U.asTypeOf(io.control.syncInc))
}
