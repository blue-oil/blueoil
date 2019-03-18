package bxb.f2a

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class TestQuantizerModule(b: Int, fWidth: Int, qWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    val qWe = Input(Bool())
    // FMem interface
    val fMemQ = Input(Vec(b, UInt(fWidth.W)))
    // QMem interface
    val qMemQ = Input(Vec(b, Vec(3, UInt(qWidth.W))))
    // AMem interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
  })
  val quantizePipeline = Module(new QuantizePipeline(b, fWidth, qWidth, aWidth))
  
  quantizePipeline.io.control.qWe := io.qWe
  quantizePipeline.io.fMemQ := io.fMemQ
  quantizePipeline.io.qMemQ := io.qMemQ
  io.aOut := quantizePipeline.io.aOut
}

class QuantizerThresTests(dut: TestQuantizerModule, b: Int, fWidth: Int, qWidth: Int, aWidth: Int ) extends PeekPokeTester(dut) {
  poke(dut.io.qWe, 1)
  for (col <- 0 until b) {
    poke(dut.io.qMemQ(col)(0), 3154.U)
    poke(dut.io.qMemQ(col)(1), 5012.U)
    poke(dut.io.qMemQ(col)(2), 8000.U)
  }
  poke(dut.io.fMemQ(0), 1000.U)
  poke(dut.io.fMemQ(1), 4000.U)
  poke(dut.io.fMemQ(2), 7000.U)
  poke(dut.io.fMemQ(3), 15000.U)
  step(1)
  poke(dut.io.qWe, 0)
  step(1)
  expect(dut.io.aOut(0), 0.U)
  expect(dut.io.aOut(1), 1.U)
  expect(dut.io.aOut(2), 2.U)
  expect(dut.io.aOut(3), 3.U)
}

object QuantizerTests {
  def main(args: Array[String]): Unit = {
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true
    val b = 4
    val memSize = 1024
    val fAddrSize = Chisel.log2Up(memSize)
    val qAddrSize = Chisel.log2Up(memSize)
    val aAddrSize = Chisel.log2Up(memSize)
    val fWidth = 16
    val qWidth = 13
    val aWidth = 2
    ok &= Driver.execute(driverArgs, () => new TestQuantizerModule(b, fWidth, qWidth, aWidth))(dut => new QuantizerThresTests(dut, b, fWidth, qWidth, aWidth))
  }
}
