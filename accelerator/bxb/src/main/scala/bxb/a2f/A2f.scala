package bxb.a2f

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.sync.{ConsumerSyncIO, ProducerSyncIO}

import bxb.util.{Util}

class A2fParameters(private val tileCountWidth: Int) extends Bundle {
  // dimensions
  val inputCCount = UInt(6.W)

  // Kernel dimensions
  val kernelVCount = UInt(2.W)
  val kernelHCount = UInt(2.W)

  // Algorithm parameters
  val tileStep = UInt(2.W)
  val tileGap = UInt(2.W)

  // Tile generation parameters
  // - should be equal to roundUp(outputHeight / tileHeight)
  val outputHCount = UInt(6.W)
  // - should be equal to roundUp(outputWidth / tileWidth)
  val outputWCount = UInt(6.W)
  // - should be equal to outputChannels / B
  val outputCCount = UInt(6.W)

  // tileHeight
  val regularTileH = UInt(tileCountWidth.W)
  // - outputHeight - (hCount - 1)  * tileHeight
  val lastTileH = UInt(tileCountWidth.W)
  // tileWidth
  val regularTileW = UInt(tileCountWidth.W)
  // - outputWidth - (wCount - 1)  * tileWidth
  val lastTileW = UInt(tileCountWidth.W)
}

object A2fParameters {
  def apply(tileCountWidth: Int) = {
    new A2fParameters(tileCountWidth)
  }
}

class A2f(b: Int, memSize: Int, aWidth: Int, fWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(memSize)
  val tileCountWidth = addrWidth
  val io = IO(new Bundle {
    val start = Input(Bool())

    // External parameters
    // - should be provided as stable signals
    val parameters = Input(A2fParameters(tileCountWidth))

    // Systolic array interface
    val aOut = Output(Vec(b, UInt(aWidth.W)))
    val evenOddOut = Output(Vec(b, UInt(1.W)))
    val accIn = Input(Vec(b, UInt(fWidth.W)))

    // AMem interface
    val amemRead = Output(Vec(b, ReadPort(addrWidth)))
    val amemQ = Input(Vec(b, UInt(aWidth.W)))

    // FMem interface
    val fmemRead = Output(Vec(b, ReadPort(addrWidth)))
    val fmemWrite = Output(Vec(b, WritePort(addrWidth, fWidth)))
    val fmemQ = Input(Vec(b, UInt(fWidth.W)))

    // Sync interface
    val aSync = ConsumerSyncIO()
    val mSync = ConsumerSyncIO()
    val fSync = ProducerSyncIO()
    
    // Status
    val statusReady = Output(Bool())
  })

  val tileAccepted = Wire(Bool())
  val tileGen = Module(new A2fTileGenerator(tileCountWidth))
  tileGen.io.start := io.start
  tileGen.io.outputHCount := io.parameters.outputHCount
  tileGen.io.outputWCount := io.parameters.outputWCount
  tileGen.io.outputCCount := io.parameters.outputCCount
  tileGen.io.regularTileH := io.parameters.regularTileH
  tileGen.io.lastTileH := io.parameters.lastTileH
  tileGen.io.regularTileW := io.parameters.regularTileW
  tileGen.io.lastTileW := io.parameters.lastTileW
  tileGen.io.tileAccepted := tileAccepted
  io.statusReady := tileGen.io.statusReady

  val sequencer = Module(new A2fSequencer(addrWidth))
  sequencer.io.inputCCount := io.parameters.inputCCount
  sequencer.io.kernelVCount := io.parameters.kernelVCount
  sequencer.io.kernelHCount := io.parameters.kernelHCount
  sequencer.io.tileVCount := tileGen.io.tileHeight
  sequencer.io.tileHCount := tileGen.io.tileWidth
  sequencer.io.tileStep := io.parameters.tileStep
  sequencer.io.tileGap := io.parameters.tileGap
  sequencer.io.tileFirst := tileGen.io.tileFirst
  sequencer.io.tileValid := tileGen.io.tileValid
  tileAccepted := sequencer.io.tileAccepted

  io.aSync.rawDec := sequencer.io.aRawDec
  sequencer.io.aRawZero := io.aSync.rawZero
  io.mSync.rawDec := sequencer.io.mRawDec
  sequencer.io.mRawZero := io.mSync.rawZero
  io.fSync.warDec := sequencer.io.fWarDec
  sequencer.io.fWarZero := io.fSync.warZero

  val pipeline = Module(new A2fPipeline(b, addrWidth, aWidth, addrWidth, fWidth))
  pipeline.io.control := sequencer.io.control
  io.aOut := pipeline.io.aOut
  io.evenOddOut := pipeline.io.evenOddOut
  pipeline.io.accIn := io.accIn
  io.amemRead := pipeline.io.amemRead
  pipeline.io.amemQ := io.amemQ
  io.fmemRead := pipeline.io.fmemRead
  pipeline.io.fmemQ := io.fmemQ
  io.fmemWrite := pipeline.io.fmemWrite

  io.aSync.warInc := pipeline.io.syncInc.aWar
  io.mSync.warInc := pipeline.io.syncInc.mWar
  io.fSync.rawInc := pipeline.io.syncInc.fRaw
}

object A2f {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new A2f(3, 1024, 2, 16)))
  }
}
