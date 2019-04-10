package bxb.f2a

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.util.{Util}

class QuantizeBlock(fWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    val writeEnable = Input(Bool())
    val qMemQ = Input(UInt(40.W))
    val fMemQ = Input(UInt(fWidth.W))
    val aOut = Output(UInt(aWidth.W))
  })
  val th0 = RegInit(0.U(13.W))
  val th1 = RegInit(0.U(13.W))
  val th2 = RegInit(0.U(13.W))
  io.aOut := 0.U
  when(io.writeEnable === true.B) {
    when(io.qMemQ(39) === 1.U) {
      th2 := io.qMemQ(38,26)
      th0 := io.qMemQ(12,0)
    }.otherwise {
      th0 := io.qMemQ(38,26)
      th2 := io.qMemQ(12,0)
    }
      th1 := io.qMemQ(25,13)
  }.otherwise {
    when(io.fMemQ > th2) {
      io.aOut := 3.U
    }.elsewhen(io.fMemQ > th1) {
      io.aOut := 2.U
    }.elsewhen(io.fMemQ > th0) {
      io.aOut := 1.U
    }.otherwise {
      io.aOut := 0.U
    }
  }
}

class QuantizePipeline(b: Int, fWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    // from controller
    val control = Input(F2aControl()) 
    // FMem interface
    val fMemQ = Input(Vec(b, UInt(fWidth.W)))
    // QMem interface
    val qMemQ = Input(Vec(b, UInt(40.W)))
    // AMem interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
  })
  
  val pipeline = Seq.fill(b){Module(new QuantizeBlock(fWidth, aWidth))}
  for (col <- 0 until b){
    pipeline(col).io.qMemQ := io.qMemQ(col)
    pipeline(col).io.writeEnable := io.control.qWe
    pipeline(col).io.fMemQ := io.fMemQ(col)
    io.aOut(col) := pipeline(col).io.aOut
  }
}
