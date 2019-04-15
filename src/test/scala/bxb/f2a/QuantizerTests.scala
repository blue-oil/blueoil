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
  step(2)
  for (col <- 0 until b) {
    val qInputsSorted = qInputs(col).sorted
    if (qSign(col) == 0L) {
      if (fInputs(col) < qInputsSorted(0)) {
        expect(dut.io.aOut(col), 0)
      } else if (fInputs(col) < qInputsSorted(1)) {
        expect(dut.io.aOut(col), 1)
      } else if (fInputs(col) < qInputsSorted(2)) {
        expect(dut.io.aOut(col), 2)
      } else {
        expect(dut.io.aOut(col), 3)
      }
    } else {
      if (fInputs(col) > qInputsSorted(0)) {
        expect(dut.io.aOut(col), 0)
      } else if (fInputs(col) > qInputsSorted(1)) {
        expect(dut.io.aOut(col), 1)
      } else if (fInputs(col) > qInputsSorted(2)) {
        expect(dut.io.aOut(col), 2)
      } else {
        expect(dut.io.aOut(col), 3)
      }
    }
  }
}

class QuantizerTests(dut: QuantizeBlock) extends PeekPokeTester(dut) {
  val reference = List(
    (14,  35, 387,  0, 198, 2),
    (-20, -9, -5,   0, -8,  2),
    (7,   59, 144,  0, 7,   1),

    (58, -76, -211, 1, -91, 2),
    (58, -76, -211, 1, 43,  1),
    (58, -76, -211, 1, 73,  0),
    (58, -76, -211, 1, 47,  1)
  )
  val mask = (0x1 << 13) - 1
  for ((th0, th1, th2, sign, value, result) <- reference) {
    poke(dut.io.writeEnable, 1)
    poke(dut.io.qMemQ, (BigInt(sign) << (3 * 13))
      | (BigInt(th2 & mask) << (2 * 13))
      | (BigInt(th1 & mask) << (1 * 13))
      | (BigInt(th0 & mask) << (0 * 13)))
    step(1)
    poke(dut.io.writeEnable, false)
    poke(dut.io.fMemQ, value & 0xFFFF)
    step(1)
    expect(dut.io.aOut, result)
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
    ok &= Driver.execute(driverArgs, () => new QuantizeBlock(fWidth, aWidth))(dut => new QuantizerTests(dut))
    if (!ok) {
      println("Quantizer Tests Fail")
      return;
    }
    ok &= Driver.execute(driverArgs, () => new TestQuantizerModule(b, fWidth, aWidth))(dut => new QuantizerThresTests(dut, b, fWidth, aWidth))
    if (!ok) {
      println("Quantization Pipeline Tests Fail")
      return;
    }
  }
}
