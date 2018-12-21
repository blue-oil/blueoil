package bxb.a2f

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.collection._

import bxb.memory.{TwoBlockMemArray, ReadPort, WritePort}

class TestAccumulationModule(b: Int, fmemSize: Int, accWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(fmemSize)
  val io = IO(new Bundle {
    // FMem read-only nterface
    val accRead = Input(Vec(b, ReadPort(addrWidth)))
    val accQ = Output(Vec(b, UInt(accWidth.W)))
    // Sustolic array interface
    val accIn = Input(Vec(b, UInt(accWidth.W)))
    // Accumulation pipeline signals
    val accumulate = Input(Bool())
    val writeEnable = Input(Bool())
    val address = Input(UInt(addrWidth.W))
  })
  val fmem = Module(new TwoBlockMemArray(b, fmemSize, accWidth))
  val accPipeline = Module(new AccumulationPipeline(b, addrWidth, accWidth))
  accPipeline.io.accumulate := io.accumulate
  accPipeline.io.writeEnable := io.writeEnable
  accPipeline.io.address := io.address
  accPipeline.io.accIn := io.accIn
  fmem.io.readA := accPipeline.io.memRead
  fmem.io.writeA := accPipeline.io.memWrite
  accPipeline.io.memQ := fmem.io.qA
  fmem.io.readB := io.accRead
  io.accQ := fmem.io.qB
}

class AccumulationPipelineTests(dut: TestAccumulationModule, b: Int, fmemSize: Int, accWidth: Int)
    extends PeekPokeTester(dut) {
  val n = 2
  val inputRows = n * fmemSize
  val inputs = Seq.fill(inputRows, b)(scala.util.Random.nextInt(0x1 << accWidth))
  val expected = mutable.Seq.fill(fmemSize, b)(0)
  // emulate 16-bit overflowing
  val addMask = (0x1 << accWidth) - 1
  for (row <- 0 until inputRows) {
    for (col <- 0 until b) {
      expected(row % fmemSize)(col) = (expected(row % fmemSize)(col) + inputs(row)(col)) & addMask
    }
  }
  val colDelay = 0 until b
  for (col <- 0 until b) {
    poke(dut.io.accRead(col).enable, false)
  }
  for (t <- 0 until (inputRows + b)) {
    for (col <- 0 until b) {
      if (t >= colDelay(col) && t - colDelay(col) < inputRows) {
        poke(dut.io.accIn(col), inputs(t - colDelay(col))(col))
      }
    }
    if (t < inputRows) {
      // just write first fmemSize values instead of accumulating
      poke(dut.io.accumulate, t >= fmemSize)
      poke(dut.io.writeEnable, true.B)
      poke(dut.io.address, t % fmemSize)
    }
    else {
      poke(dut.io.writeEnable, false.B)
    }
    step(1)
  }
  step(1)
  for (row <- 0 until fmemSize) {
    for (col <- 0 until b) {
      poke(dut.io.accRead(col).enable, true)
      poke(dut.io.accRead(col).addr, row)
    }
    step(1)
    for (col <- 0 until b) {
      expect(dut.io.accQ(col), expected(row)(col))
    }
  }
}

object AccumulationPipelineTests {
  def main(args: Array[String]): Unit = { 
    val b = 8
    val fmemSize = 512
    val ok = Driver(() => new TestAccumulationModule(b, fmemSize, 16))(c => new AccumulationPipelineTests(c, b, fmemSize, 16))
    if (!ok && args(0) != "noexit")
      System.exit(1)
  }
}
