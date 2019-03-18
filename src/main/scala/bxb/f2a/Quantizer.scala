package bxb.f2a

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.util.{Util}

class QuantizeBlock(fWidth: Int, qWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    val writeEnable = Input(Bool())
    val qMemQ = Input(Vec(3, UInt(qWidth.W)))
    val fMemQ = Input(UInt(fWidth.W))
    val aOut = Output(UInt(aWidth.W))
  })
  val thres = RegInit(0.U.asTypeOf(io.qMemQ))
  io.aOut := 0.U
  when(io.writeEnable === true.B) {
    thres := io.qMemQ
  }
    
  when(io.fMemQ > thres(2)) {
    io.aOut := 3.U
  }.elsewhen(io.fMemQ > thres(1)) {
    io.aOut := 2.U
  }.elsewhen(io.fMemQ > thres(0)) {
    io.aOut := 1.U
  }.otherwise {
    io.aOut := 0.U
  }
}

class QuantizePipeline(b: Int, fWidth: Int, qWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    // from controller
    val control = Input(F2aControl()) 
    // FMem interface
    val fMemQ = Input(Vec(b, UInt(fWidth.W)))
    // QMem interface
    val qMemQ = Input(Vec(b, Vec(3, UInt(qWidth.W))))
    // AMem interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
  })
  
  val pipeline = Seq.fill(b){Module(new QuantizeBlock(fWidth, qWidth, aWidth))}
  for (col <- 0 until b){
    pipeline(col).io.qMemQ := io.qMemQ(col)
    pipeline(col).io.writeEnable := io.control.qWe
    pipeline(col).io.fMemQ := io.fMemQ(col)
    io.aOut(col) := pipeline(col).io.aOut
  }
}
