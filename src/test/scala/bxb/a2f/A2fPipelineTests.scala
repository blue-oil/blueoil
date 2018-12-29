package bxb.a2f

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.collection._

import bxb.array.{MacArray}
import bxb.test.{TestUtil}
import bxb.memory.{ReadPort, WritePort, MemArray, TwoBlockMemArray}

// Dummy top level module to test integration between fmem, amem, a2f pipeline and mac array
class TestA2fModule(b: Int, memSize: Int, aWidth: Int, fWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(memSize)
  val io = IO(new Bundle {
    // Pipeline signals
    val control = Input(A2fControl(addrWidth, addrWidth))
    // Systolic array weight loading interface
    val mIn = Input(Vec(b, Vec(2, UInt(1.W))))
    val mWe = Input(Vec(2, Bool()))
    val evenOddIn = Input(Vec(b, UInt(1.W)))
    // AMem test interface
    val amemWrite = Input(Vec(b, WritePort(addrWidth, aWidth)))
    // FMem test interface
    val fmemRead = Input(Vec(b, ReadPort(addrWidth)))
    val fmemQ = Output(Vec(b, UInt(fWidth.W)))
  })
  // TODO: change parameter order of systolic array to match [b, a, f] order
  val macArray = Module(new MacArray(b, fWidth, aWidth))
  macArray.io.mIn := io.mIn
  macArray.io.mWe := io.mWe
  macArray.io.evenOddIn := io.evenOddIn
  val amem = Module(new MemArray(b, memSize, aWidth))
  amem.io.write := io.amemWrite
  val fmem = Module(new TwoBlockMemArray(b, memSize, fWidth))
  fmem.io.readB := io.fmemRead
  io.fmemQ := fmem.io.qB
  val a2fPipeline = Module(new A2fPipeline(b, addrWidth, aWidth, addrWidth, fWidth))
  a2fPipeline.io.control := io.control
  macArray.io.aIn := a2fPipeline.io.aOut
  a2fPipeline.io.accIn := macArray.io.accOut
  amem.io.read := a2fPipeline.io.amemRead
  a2fPipeline.io.amemQ := amem.io.q
  fmem.io.readA := a2fPipeline.io.fmemRead
  a2fPipeline.io.fmemQ := fmem.io.qA
  fmem.io.writeA := a2fPipeline.io.fmemWrite
}

class A2fPipelineTests(dut: TestA2fModule, b: Int, memSize: Int) extends PeekPokeTester(dut) {
  val n = memSize / 2

  def fillM(pane: Int, m: Seq[Seq[Int]]) = {
    poke(dut.io.mWe(pane), true)
    for (col <- 0 until b) {
      for (row <- 0 until b) {
        poke(dut.io.mIn(row)(pane), m(row)(col))
      }
      step(1)
    }
    poke(dut.io.mWe(pane), false)
  }

  def fillAMem(offset: Int, input: Seq[Seq[Int]]) = {
    for (row <- 0 until n) {
      for (col <- 0 until b) {
        poke(dut.io.amemWrite(col).addr, offset + row)
        poke(dut.io.amemWrite(col).data, input(row)(col))
        poke(dut.io.amemWrite(col).enable, true)
      }
      step(1)
    }
    for (col <- 0 until b) {
        poke(dut.io.amemWrite(col).enable, false)
    }
  }

  val weights = Seq.fill(b, b)(scala.util.Random.nextInt(2))
  val inputs = Seq.fill(2, n, b)(scala.util.Random.nextInt(4))
  val outputs = TestUtil.matAdd(TestUtil.matMul(inputs(0), weights), TestUtil.matMul(inputs(1), weights))

  // calculate outputs = inputs0 * weights + inputs1 * weights
  // where
  //  inputs0, inputs1 - are NxB matrices of 2 bit activations
  //  weights - is BxB matrix of 1 bit weights
  //  outputs - is NxB matrix of 16 bit features
  val pane = 0
  for (col <- 0 until b) {
    poke(dut.io.evenOddIn(col), pane)
  }
  poke(dut.io.control.writeEnable, false)
  for (col <- 0 until b) {
    poke(dut.io.fmemRead(col).enable, false)
  }
  fillM(pane, weights)
  for (i <- 0 until 2) {
    fillAMem(i * n, inputs(i))
  }
  for (row <- 0 until 2 * n) {
    poke(dut.io.control.accumulate, row >= n)
    poke(dut.io.control.writeEnable, true)
    poke(dut.io.control.aAddr, row)
    poke(dut.io.control.fAddr, row % n)
    step(1)
  }
  poke(dut.io.control.writeEnable, false)
  step(2 * b)
  for (row <- 0 until n) {
    for (col <- 0 until b) {
      poke(dut.io.fmemRead(col).addr, row)
      poke(dut.io.fmemRead(col).enable, true)
    }
    step(1)
    for (col <- 0 until b) {
      expect(dut.io.fmemQ(col), outputs(row)(col))
    }
  }
}

object A2fPipelineTests {
  def main(args: Array[String]): Unit = {
    val b = 16;
    val n = 1024;
    val ok = Driver(() => new TestA2fModule(b, n, 2, 16))(c => new A2fPipelineTests(c, b, n))
    if (!ok && args(0) != "noexit")
      System.exit(1)
  }
}
