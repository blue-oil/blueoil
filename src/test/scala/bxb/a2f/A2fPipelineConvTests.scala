package bxb.a2f

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.collection._

import bxb.array.{MacArray}
import bxb.test.{TestUtil}
import bxb.memory.{ReadPort, WritePort, MemArray, TwoBlockMemArray}

class ReferenceConv(val b: Int, val tileHeight: Int, val tileWidth: Int) {
  val inputChannels = b
  val outputChannels = b
  val input = Seq.fill(tileHeight, tileWidth, inputChannels)(scala.util.Random.nextInt(4))
  val kernelHeight = 3
  val kernelWidth = 3
  val kernels = Seq.fill(kernelHeight, kernelWidth, outputChannels, inputChannels)(scala.util.Random.nextInt(2))
  val decodedKernels = kernels.map(_.map(_.map(_.map({x => if (x == 0) 1 else -1}))))
  val outputHeight = tileHeight - 2
  val outputWidth = tileWidth - 2
  val output = mutable.Seq.fill(outputHeight, outputWidth, outputChannels)(0)
  for (oy <- 0 until outputHeight) {
    for (ox <- 0 until outputWidth) {
      for (oc <- 0 until outputChannels) {
        var pixel = 0
        // iterating over input sector convolved
        // to get output pixel
        for (ky <- 0 until 3) {
          for (kx <- 0 until 3) {
            for (kc <- 0 until inputChannels) {
              pixel += input(oy + ky)(ox + kx)(kc) * decodedKernels(ky)(kx)(oc)(kc)
            }
          }
        }
        output(oy)(ox)(oc) = pixel
      }
    }
  }
}

class TestA2fConvModule(b: Int, memSize: Int, aWidth: Int, fWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(memSize)
  val io = IO(new Bundle {
    // Systolic array weight loading interface
    val mIn = Input(Vec(b, Vec(2, UInt(1.W))))
    val mWe = Input(Vec(2, Bool()))
    // AMem test interface
    val amemWrite = Input(Vec(b, WritePort(addrWidth, aWidth)))
    // FMem test interface
    val fmemRead = Input(Vec(b, ReadPort(addrWidth)))
    val fmemQ = Output(Vec(b, UInt(fWidth.W)))
    // Sequencer interface
    val kernelVCount = Input(UInt(2.W))
    val kernelHCount = Input(UInt(2.W))
    val tileVCount = Input(UInt(addrWidth.W))
    val tileHCount = Input(UInt(addrWidth.W))
    val tileStep = Input(UInt(2.W))
    val tileGap = Input(UInt(2.W))
    val tileOffset = Input(UInt(addrWidth.W))
    val tileOffsetValid = Input(Bool())
    // Sequencer Sync interface
    // XXX: This will be connected to semaphores eventually
    val mRawZero = Input(Bool())
    val controlValid = Output(Bool())
  })
  val macArray = Module(new MacArray(b, fWidth, aWidth))
  macArray.io.mIn := io.mIn
  macArray.io.mWe := io.mWe
  val amem = Module(new MemArray(b, memSize, aWidth))
  amem.io.write := io.amemWrite
  val fmem = Module(new TwoBlockMemArray(b, memSize, fWidth))
  fmem.io.readB := io.fmemRead
  io.fmemQ := fmem.io.qB
  val a2fSequencer = Module(new A2fSequencer(addrWidth))
  a2fSequencer.io.kernelVCount := io.kernelVCount
  a2fSequencer.io.kernelHCount := io.kernelHCount
  a2fSequencer.io.tileVCount := io.tileVCount
  a2fSequencer.io.tileHCount := io.tileHCount
  a2fSequencer.io.tileStep := io.tileStep 
  a2fSequencer.io.tileGap := io.tileGap 
  a2fSequencer.io.tileOffset := io.tileOffset 
  a2fSequencer.io.tileOffsetValid := io.tileOffsetValid 
  io.controlValid := a2fSequencer.io.controlValid
  a2fSequencer.io.aRawZero := false.B
  a2fSequencer.io.mRawZero := io.mRawZero
  val a2fPipeline = Module(new A2fPipeline(b, addrWidth, aWidth, addrWidth, fWidth))
  a2fPipeline.io.control := a2fSequencer.io.control
  macArray.io.aIn := a2fPipeline.io.aOut
  macArray.io.evenOddIn := a2fPipeline.io.evenOddOut
  a2fPipeline.io.accIn := macArray.io.accOut
  amem.io.read := a2fPipeline.io.amemRead
  a2fPipeline.io.amemQ := amem.io.q
  fmem.io.readA := a2fPipeline.io.fmemRead
  a2fPipeline.io.fmemQ := fmem.io.qA
  fmem.io.writeA := a2fPipeline.io.fmemWrite
}

class A2fPipelineConvTests(dut: TestA2fConvModule, b: Int, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  def dumpRef(ref: ReferenceConv): Unit = {
    for (c <- 0 until ref.inputChannels) {
      println(f"ref.input channel=${c}")
      for (y <- 0 until ref.tileHeight) {
        for (x <- 0 until ref.tileWidth) {
          print(f" ${ref.input(y)(x)(c)}")
        }
        print("\n")
      }
      println(f"ref.decodedKernels[0] channel=${c}")
      for (y <- 0 until ref.kernelHeight) {
        for (x <- 0 until ref.kernelWidth) {
          print(f" ${ref.decodedKernels(y)(x)(0)(c)}")
        }
        print("\n")
      }
    }
    println("ref.output channel=0")
    for (y <- 0 until ref.outputHeight) {
      for (x <- 0 until ref.outputWidth) {
        print(f" ${ref.output(y)(x)(0)}")
      }
      print("\n")
    }
  }

  val ref = new ReferenceConv(b, tileHeight, tileWidth)

  dumpRef(ref)

  def fillM(pane: Int, m: Seq[Seq[Int]]) = {
    poke(dut.io.mWe(pane), true)
    for (kernel <- 0 until b) {
      for (channel <- 0 until b) {
        poke(dut.io.mIn(channel)(pane), m(kernel)(channel))
      }
      step(1)
    }
    poke(dut.io.mWe(pane), false)
  }

  def loadTileToAMem() = {
    for (y <- 0 until ref.tileHeight) {
      for (x <- 0 until ref.tileWidth) {
        val addr = y * ref.tileWidth + x
        for (channel <- 0 until b) {
          poke(dut.io.amemWrite(channel).addr, addr)
          poke(dut.io.amemWrite(channel).data, ref.input(y)(x)(channel))
          poke(dut.io.amemWrite(channel).enable, true)
        }
        step(1)
      }
    }
    for (channel <- 0 until b) {
        poke(dut.io.amemWrite(channel).enable, false)
    }
  }

  for (col <- 0 until b) {
    poke(dut.io.fmemRead(col).enable, false)
  }
  poke(dut.io.mRawZero, true)
  poke(dut.io.tileOffsetValid, false)

  loadTileToAMem()
  // Sequencer starts its even/odd sequence from 0 as well
  var evenOdd = 0
  for (ky <- 0 until ref.kernelHeight) {
    for (kx <- 0 until ref.kernelWidth) {
      // Load next set of kernels
      fillM(evenOdd, ref.kernels(ky)(kx))
      poke(dut.io.kernelVCount, ref.kernelHeight)
      poke(dut.io.kernelHCount, ref.kernelWidth)
      poke(dut.io.tileVCount, ref.outputHeight)
      poke(dut.io.tileHCount, ref.outputWidth)
      poke(dut.io.tileStep, 1)
      poke(dut.io.tileGap, 3)
      poke(dut.io.tileOffset, 0)
      poke(dut.io.mRawZero, false)
      poke(dut.io.tileOffsetValid, true)
      while (peek(dut.io.controlValid) == 0) {
        step(1)
      }
      for (i <- 0 until (ref.outputHeight * ref.outputWidth)) {
        step(1)
      }
      poke(dut.io.mRawZero, true)
      poke(dut.io.tileOffsetValid, false)
      evenOdd = evenOdd ^ 0x1
    }
  }
  // Wait long enough to be sure all computations completed
  for (i <- 0 until (2 * b)) {
    step(1)
  }
  // Assume results are 16 bit wide
  val mask = (0x1 << 16) - 1
  for (oy <- 0 until ref.outputHeight) {
    for (ox <- 0 until ref.outputWidth) {
      val addr = oy * ref.outputWidth + ox 
      for (channel <- 0 until b) {
        poke(dut.io.fmemRead(channel).addr, addr)
        poke(dut.io.fmemRead(channel).enable, true)
      }
      step(1)
      for (channel <- 0 until b) {
        expect(dut.io.fmemQ(channel), ref.output(oy)(ox)(channel) & mask)
      }
    }
  }
}

object A2fPipelineConvTests {
  def main(args: Array[String]): Unit = {
    val b = 3
    val memSize = 1024
    val tileHeight = 16
    val tileWidth = 32
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "false", "--test-seed", "1547278051217")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    val ok = Driver.execute(driverArgs, () => new TestA2fConvModule(b, memSize, 2, 16))(c => new A2fPipelineConvTests(c, b, tileHeight, tileWidth))
    if (!ok && !args.contains("noexit"))
      System.exit(1)
  }
}
