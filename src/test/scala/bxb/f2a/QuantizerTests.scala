package bxb.f2a
import scala.collection._
import chisel3._
import chisel3.util._
import scala.math._
import chisel3.iotesters.{PeekPokeTester, Driver}

class TestQuantizerModule(b: Int, fWidth: Int, aWidth: Int) extends Module {
  val io = IO(new Bundle {
    val qWe = Input(Bool())
    // FMem interface
    val fMemQ = Input(Vec(b, UInt(fWidth.W)))
    // QMem interface
    val qMemQ = Input(Vec(b, UInt(40.W)))
    // AMem interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
  })
  val quantizePipeline = Module(new QuantizePipeline(b, fWidth, aWidth))
  
  quantizePipeline.io.qWe := io.qWe
  quantizePipeline.io.fMemQ := io.fMemQ
  quantizePipeline.io.qMemQ := io.qMemQ
  io.aOut := quantizePipeline.io.aOut
}

class QuantizerThresTests(dut: TestQuantizerModule, b: Int, fWidth: Int, aWidth: Int ) extends PeekPokeTester(dut) {
  val qInputs = Seq.fill(b,3)(scala.util.Random.nextInt(8192)) // 13bit
  val qSign = Seq.fill(b)(scala.util.Random.nextInt(2))
  val fInputs = Seq.fill(b)(scala.util.Random.nextInt(8192))
  val aOutputs = Seq.fill(b)(0.U(aWidth.W))
  
  poke(dut.io.qWe, 1)
  for (col <- 0 until b) {
    val qInputsSorted = qInputs(col).sorted
    val th2 = qInputsSorted(2)
    val th1 = qInputsSorted(1)
    val th0 = qInputsSorted(0)
    val th = (((th2.toLong << 13) << 13) + (th1.toLong << 13) + th0.toLong) + (qSign(col).toLong << 39)
    
    poke(dut.io.qMemQ(col), th)
    poke(dut.io.fMemQ(col), fInputs(col))
  }
  step(1)
  poke(dut.io.qWe, 0)
  step(1)
  for (col <- 0 until b) {
    val qInputsSorted = qInputs(col).sorted
    if (qSign(col) == 1L) {
      if (fInputs(col) > qInputsSorted(2)) {
        expect(dut.io.aOut(col), 3.U)
      } else if (fInputs(col) > qInputsSorted(1)) {
        expect(dut.io.aOut(col), 2.U)
      } else if (fInputs(col) > qInputsSorted(0)) {
        expect(dut.io.aOut(col), 1.U)
      } else {
        expect(dut.io.aOut(col), 0.U)
      }
    } else {
      if (fInputs(col) > qInputsSorted(0)) {
        expect(dut.io.aOut(col), 3.U)
      } else if (fInputs(col) > qInputsSorted(1)) {
        expect(dut.io.aOut(col), 2.U)
      } else if (fInputs(col) > qInputsSorted(2)) {
        expect(dut.io.aOut(col), 1.U)
      } else {
        expect(dut.io.aOut(col), 0.U)
      }
    }
  }
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
    val aWidth = 2
    ok &= Driver.execute(driverArgs, () => new TestQuantizerModule(b, fWidth, aWidth))(dut => new QuantizerThresTests(dut, b, fWidth, aWidth))
  }
}
