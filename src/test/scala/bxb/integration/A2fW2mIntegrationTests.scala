package bxb.integration

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.collection._

import bxb.a2f.{A2fSequencer, A2fPipeline}
import bxb.array.{MacArray}
import bxb.test.{TestUtil}
import bxb.memory.{ReadPort, WritePort, PackedWritePort, MemArray, TwoBlockMemArray, PackedBlockRam}
import bxb.sync.{SemaphorePair}
import bxb.w2m.{W2mSequencer, W2mPipeline}

class ReferenceConv3x3(val b: Int, val tileHeight: Int, val tileWidth: Int) {
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

class TestA2fW2mIntegrationModule(b: Int, amemSize: Int, aWidth: Int, fmemSize: Int, fWidth: Int, wmemSize: Int) extends Module {
  val aAddrWidth = Chisel.log2Up(amemSize)
  val fAddrWidth = Chisel.log2Up(fmemSize)
  val wAddrWidth = Chisel.log2Up(wmemSize)
  val io = IO(new Bundle {
    // AMem test interface
    val amemWrite = Input(Vec(b, WritePort(aAddrWidth, aWidth)))
    // FMem test interface
    val fmemRead = Input(Vec(b, ReadPort(fAddrWidth)))
    val fmemQ = Output(Vec(b, UInt(fWidth.W)))
    // WMem test interface
    val wmemWrite = Input(PackedWritePort(wAddrWidth, b, 1))
    // Sequencer interface
    val kernelVCount = Input(UInt(2.W))
    val kernelHCount = Input(UInt(2.W))
    val tileVCount = Input(UInt(aAddrWidth.W))
    val tileHCount = Input(UInt(aAddrWidth.W))
    val tileStep = Input(UInt(3.W))
    val tileGap = Input(UInt(2.W))
    val tileValid = Input(Bool())
    // AMem sync test interface
    val aWarZero = Output(Bool())
    val aWarDec = Input(Bool())
    val aRawInc = Input(Bool())
    // WMem sync test interface
    val wWarZero = Output(Bool())
    val wWarDec = Input(Bool())
    val wRawInc = Input(Bool())
    // FMem sync test interface
    val fRawZero = Output(Bool())
    val fRawDec = Input(Bool())
    val fWarInc = Input(Bool())
  })

  // preset war count to 2 as we want to load both panes ASAP
  val mSemaPair = Module(new SemaphorePair(2, 0, 2))
  val macArray = Module(new MacArray(b, fWidth, aWidth))

  // preset war count to 1 as we want to load one tile and stop after it
  val aSemaPair = Module(new SemaphorePair(2, 0, 1))
  io.aWarZero := aSemaPair.io.producer.warZero
  aSemaPair.io.producer.warDec := io.aWarDec
  aSemaPair.io.producer.rawInc := io.aRawInc
  val amem = Module(new MemArray(b, fmemSize, aWidth))
  amem.io.write := io.amemWrite

  // preset war count to 1 as we want to process one tile and stop after it
  val fSemaPair = Module(new SemaphorePair(2, 0, 1))
  io.fRawZero := fSemaPair.io.consumer.rawZero
  fSemaPair.io.consumer.rawDec := io.fRawDec
  fSemaPair.io.consumer.warInc := io.fWarInc
  val fmem = Module(new TwoBlockMemArray(b, fmemSize, fWidth))
  fmem.io.readB := io.fmemRead
  io.fmemQ := fmem.io.qB

  // preset war count to 9 as we want load BxBx9 kernels ASAP
  val wSemaPair = Module(new SemaphorePair(4, 0, 9))
  io.wWarZero := wSemaPair.io.producer.warZero

  wSemaPair.io.producer.warDec := io.wWarDec
  wSemaPair.io.producer.rawInc := io.wRawInc

  val wmem = Module(new PackedBlockRam(b, wmemSize, 1))
  wmem.io.write := io.wmemWrite

  val w2mSequencer = Module(new W2mSequencer(b, wAddrWidth))
  w2mSequencer.io.mWarZero := mSemaPair.io.producer.warZero
  mSemaPair.io.producer.warDec := w2mSequencer.io.mWarDec
  w2mSequencer.io.wRawZero := wSemaPair.io.consumer.rawZero
  wSemaPair.io.consumer.rawDec := w2mSequencer.io.wRawDec

  val w2mPipeline = Module(new W2mPipeline(b, wAddrWidth))
  w2mPipeline.io.control := w2mSequencer.io.control
  wmem.io.read := w2mPipeline.io.wmemRead
  w2mPipeline.io.wmemQ := wmem.io.q

  macArray.io.mIn := w2mPipeline.io.mOut
  macArray.io.mWe := w2mPipeline.io.mWe

  mSemaPair.io.producer.rawInc := w2mPipeline.io.mRawInc
  wSemaPair.io.consumer.warInc := w2mPipeline.io.wWarInc

  val a2fSequencer = Module(new A2fSequencer(aAddrWidth))
  a2fSequencer.io.kernelVCount := io.kernelVCount
  a2fSequencer.io.kernelHCount := io.kernelHCount
  a2fSequencer.io.inputCCount := 1.U
  a2fSequencer.io.tileVCount := io.tileVCount
  a2fSequencer.io.tileHCount := io.tileHCount
  a2fSequencer.io.tileStep := io.tileStep 
  a2fSequencer.io.tileGap := io.tileGap 
  a2fSequencer.io.tileValid := io.tileValid

  a2fSequencer.io.aRawZero := aSemaPair.io.consumer.rawZero
  aSemaPair.io.consumer.rawDec := a2fSequencer.io.aRawDec
  a2fSequencer.io.mRawZero := mSemaPair.io.consumer.rawZero
  mSemaPair.io.consumer.rawDec := a2fSequencer.io.mRawDec
  a2fSequencer.io.fWarZero := fSemaPair.io.producer.warZero
  fSemaPair.io.producer.warDec := a2fSequencer.io.fWarDec

  val a2fPipeline = Module(new A2fPipeline(b, aAddrWidth, aWidth, fAddrWidth, fWidth))
  a2fPipeline.io.control := a2fSequencer.io.control
  macArray.io.aIn := a2fPipeline.io.aOut
  macArray.io.evenOddIn := a2fPipeline.io.evenOddOut
  a2fPipeline.io.accIn := macArray.io.accOut
  amem.io.read := a2fPipeline.io.amemRead
  a2fPipeline.io.amemQ := amem.io.q
  fmem.io.readA := a2fPipeline.io.fmemRead
  a2fPipeline.io.fmemQ := fmem.io.qA
  fmem.io.writeA := a2fPipeline.io.fmemWrite

  aSemaPair.io.consumer.warInc := a2fPipeline.io.syncInc.aWar
  mSemaPair.io.consumer.warInc := a2fPipeline.io.syncInc.mWar
  fSemaPair.io.producer.rawInc := a2fPipeline.io.syncInc.fRaw
}

class A2fW2mIntegrationConvTests(dut: TestA2fW2mIntegrationModule, b: Int, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  def dumpRef(ref: ReferenceConv3x3): Unit = {
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

  val ref = new ReferenceConv3x3(b, tileHeight, tileWidth)

  dumpRef(ref)

  object AMemLoader {
    // Essentially it does
    // wait until ready
    // aWar -= 1
    // for (y <- 0 until ref.tileHeight)
    //   for (x <- 0 until ref.tileWidth)
    //     val addr = y * ref.tileWidth + x
    //     amem.write(addr, ref.input(y)(x))
    // aRaw += 1

    private var y = 0
    private var x = 0

    private val running = 0 // in the midle of process of writing data
    private val waiting = 0 // waiting for wWarZero to be able to start writing

    private var state = waiting

    def done = (y == ref.tileHeight)

    private def ready = (state == running) || (state == waiting && peek(dut.io.aWarZero) == 0) // ready when false

    private def writeAndAdvance() = {
      // we should decrement aWar immediately after we started to use the tile
      poke(dut.io.aWarDec, (y == 0 && x == 0))
      val addr = y * ref.tileWidth + x
      for (channel <- 0 until b) {
        poke(dut.io.amemWrite(channel).addr, addr)
        poke(dut.io.amemWrite(channel).data, ref.input(y)(x)(channel))
        poke(dut.io.amemWrite(channel).enable, true)
      }
      x += 1
      if (x == ref.tileWidth) {
        x = 0
        y += 1
      }
      // we have to notify a2f pipeline when we loaded whole tile for it
      state = if (done) waiting else running
      poke(dut.io.aRawInc, done) // increment when loaded whole tile
    }

    private def doNotWrite() = {
      for (channel <- 0 until b) {
        poke(dut.io.amemWrite(channel).enable, false)
        poke(dut.io.aRawInc, false)
        poke(dut.io.aWarDec, false)
      }
    }

    def tryNext() {
      if (done || !ready) {
        doNotWrite()
      }
      else {
        writeAndAdvance()
      }
    }

    // reset write/increment signals
    def onDone() {
      doNotWrite()
    }
  }

  object WMemLoader {
    // for (ky <- 0 until ref.kernelHeight)
    //   for (kx <- 0 until ref.kernelWidth)
    //     wait until ready
    //     wWar -= 1
    //     for (outputChannel <- 0 until b)
    //       val addr = ky * ref.kernelWidth * b + kx * b + outputChannel
    //       wmem.write(addr, ref.kernels(ky, kx, outputChannel))
    //     wRaw += 1

    private var ky = 0
    private var kx = 0
    private var outputChannel = 0

    private val running = 0 // in the midle of process of writing data
    private val waiting = 0 // waiting for wWarZero to be able to start writing

    private var state = waiting

    def done = (ky == ref.kernelHeight)

    private def ready = (state == running) || (state == waiting && peek(dut.io.wWarZero) == 0) // ready when false

    private def writeAndAdvance() = {
      // we should decrement wWar immediately after we use next bxb chunk of kernels
      poke(dut.io.wWarDec, (outputChannel == 0))
      val addr = ky * ref.kernelWidth * b + kx * b + outputChannel
      poke(dut.io.wmemWrite.addr, addr)
      poke(dut.io.wmemWrite.enable, true)
      for (inputChannel <- 0 until b) {
        poke(dut.io.wmemWrite.data(inputChannel), ref.kernels(ky)(kx)(outputChannel)(inputChannel))
      }
      outputChannel += 1
      val paneLoaded = (outputChannel == b) // one pane sized chunk of data loaded
      // we have to notify w2m pipeline each time we loaded one pane so it could start consume it
      poke(dut.io.wRawInc, paneLoaded)
      if (paneLoaded) {
        outputChannel = 0
        kx += 1
        if (kx == ref.kernelWidth) {
          kx = 0
          ky += 1
        }
      }
      state = if (paneLoaded) waiting else running
    }

    private def doNotWrite() = {
      poke(dut.io.wmemWrite.enable, false)
      poke(dut.io.wRawInc, false)
      poke(dut.io.wWarDec, false)
    }

    def tryNext() {
      if (done || !ready) {
        doNotWrite()
      }
      else {
        writeAndAdvance()
      }
    }

    // reset write/increment signals
    def onDone() {
      doNotWrite()
    }
  }

  poke(dut.io.kernelVCount, ref.kernelHeight)
  poke(dut.io.kernelHCount, ref.kernelWidth)
  poke(dut.io.tileVCount, ref.outputHeight)
  poke(dut.io.tileHCount, ref.outputWidth)
  poke(dut.io.tileStep, 1)
  poke(dut.io.tileGap, 3)
  poke(dut.io.tileValid, true)

  val computationTime = (ref.tileHeight * ref.tileWidth * ref.kernelHeight * ref.kernelWidth + 2 * b)
  val amemLoadTime = (ref.tileHeight * ref.tileWidth)
  val fmemLoadTime = (ref.kernelHeight * ref.kernelWidth * b)
  // just make it long enough: time of all tasks as if they executed seqyentially by 5
  val timeout = t + 5 * (computationTime + amemLoadTime + fmemLoadTime)

  // Interleave loading and computations
  while ((!AMemLoader.done || !WMemLoader.done) && (t < timeout)) {
    AMemLoader.tryNext()
    WMemLoader.tryNext()
    step(1)
    // t += 1
  }
  AMemLoader.onDone()
  WMemLoader.onDone()

  // Wait until completion
  while (peek(dut.io.fRawZero) != 0 && t < timeout) {
    step(1)
  }

  if (t == timeout) {
    println("A2fW2mIntegrationConvTests: TIMEOUT")
  }
  // // Wait long enough to be sure all computations completed
  // for (i <- 0 until (ref.tileHeight * ref.tileWidth * ref.kernelHeight * ref.kernelWidth + 2 * b)) {
  //   step(1)
  // }
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

object A2fW2mIntegrationConvTests {
  def main(args: Array[String]): Unit = {
    val b = 8
    val memSize = 1024
    val tileHeight = 16
    val tileWidth = 32
    //val verilatorArgs = Array("--backend-name", "verilator")
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "false", "--test-seed", "1547278051217")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    val ok = Driver.execute(driverArgs, () => new TestA2fW2mIntegrationModule(b, memSize, 2, memSize, 16, memSize))(c => new A2fW2mIntegrationConvTests(c, b, tileHeight, tileWidth))
    if (!ok && !args.contains("noexit"))
      System.exit(1)
  }
}
