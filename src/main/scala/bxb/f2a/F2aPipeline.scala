package bxb.f2a

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.util.{Util}

class F2aPipeline(b: Int, fWidth: Int, qWidth: Int, aWidth: Int, addrWidth: Int) extends Module {
  val io = IO(new Bundle{
    val control = Input(F2aControl())
    val fMemQ = Input(Vec(b, UInt(fWidth.W)))
    val qMemQ = Input(Vec(b, UInt(40.W)))
    val amemWriteAddr = Input(UInt(addrWidth.W))
    val amemWrite = Output(Vec(b, WritePort(addrWidth, aWidth)))
    val writeEnable = Input(Bool())
  })
  val quantizer = Module(new QuantizePipeline(b, fWidth, aWidth))
  val addrBuf = RegNext(io.amemWriteAddr)
  val writeEnableBuf = RegNext(io.writeEnable, init=false.B)

  quantizer.io.control := io.control
  quantizer.io.fMemQ   := io.fMemQ  
  quantizer.io.qMemQ   := io.qMemQ  
  for (col <- 0 until b) {
    io.amemWrite(col).data := quantizer.io.aOut(col)
    io.amemWrite(col).addr := addrBuf
    io.amemWrite(col).enable := writeEnableBuf
  }
}
