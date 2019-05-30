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
  val th0 = Reg(SInt(13.W))
  val th1 = Reg(SInt(13.W))
  val th2 = Reg(SInt(13.W))
  val sign = Reg(Bool())

  when(io.writeEnable === true.B) {
    th2 := io.qMemQ(38,26).asSInt
    th1 := io.qMemQ(25,13).asSInt
    th0 := io.qMemQ(12,0).asSInt
    sign := io.qMemQ(39)
  }

  val value = io.fMemQ.asSInt
  when(~sign) {
    when(RegNext(value < th0)) {
      io.aOut := 0.U
    }.elsewhen(RegNext(value < th1)) {
      io.aOut := 1.U
    }.elsewhen(RegNext(value < th2)) {
      io.aOut := 2.U
    }.otherwise {
      io.aOut := 3.U
    }
  }.otherwise {
    when(RegNext(value > th0)) {
      io.aOut := 0.U
    }.elsewhen(RegNext(value > th1)) {
      io.aOut := 1.U
    }.elsewhen(RegNext(value > th2)) {
      io.aOut := 2.U
    }.otherwise {
      io.aOut := 3.U
    }
  }
}

class QuantizePipeline(b: Int, fWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    // from controller
    val qWe = Input(Bool())
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
    pipeline(col).io.writeEnable := io.qWe
    pipeline(col).io.fMemQ := io.fMemQ(col)
    io.aOut(col) := pipeline(col).io.aOut
  }
}
