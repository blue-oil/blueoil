package bxb.f2a

import chisel3._

import bxb.a2f.{A2fTileGenerator}
import bxb.memory.{ReadPort, WritePort}
import bxb.sync.{ConsumerSyncIO, ProducerSyncIO}

import bxb.util.{Util}

class F2a(b: Int, dataMemSize: Int, qmemSize: Int, aWidth: Int, fWidth: Int) extends Module {
  val dataAddrWidth = Chisel.log2Up(dataMemSize)
  val qAddrWidth = Chisel.log2Up(qmemSize)
  val tileCountWidth = dataAddrWidth
  val io = IO(new Bundle {
    val start = Input(Bool())

    // Tile generation parameters
    // - should be equal to roundUp(outputHeight / tileHeight)
    val outputHCount = Input(UInt(6.W))
    // - should be equal to roundUp(outputWidth / tileWidth)
    val outputWCount = Input(UInt(6.W))
    // - should be equal to outputChannels / B
    val outputCCount = Input(UInt(6.W))

    // tileHeight
    val regularTileH = Input(UInt(tileCountWidth.W))
    // - outputHeight - (hCount - 1)  * tileHeight
    val lastTileH = Input(UInt(tileCountWidth.W))
    // tileWidth
    val regularTileW = Input(UInt(tileCountWidth.W))
    // - outputWidth - (wCount - 1)  * tileWidth
    val lastTileW = Input(UInt(tileCountWidth.W))

    // AMem interface
    val amemWrite = Output(Vec(b, WritePort(dataAddrWidth, aWidth)))

    // FMem interface
    val fmemRead = Output(Vec(b, ReadPort(dataAddrWidth)))
    val fmemQ = Input(Vec(b, UInt(fWidth.W)))

    // QMem interface
    val qmemRead = Output(ReadPort(qAddrWidth))
    val qmemQ = Input(Vec(b, UInt(40.W)))

    // Sync interface
    val aSync = ProducerSyncIO()
    val qSync = ConsumerSyncIO()
    val fSync = ConsumerSyncIO()

    // Status
    val statusReady = Output(Bool())
  })

  val tileAccepted = Wire(Bool())
  val tileGen = Module(new A2fTileGenerator(tileCountWidth))
  tileGen.io.start := io.start
  tileGen.io.outputHCount := io.outputHCount
  tileGen.io.outputWCount := io.outputWCount
  tileGen.io.outputCCount := io.outputCCount
  tileGen.io.regularTileH := io.regularTileH
  tileGen.io.lastTileH := io.lastTileH
  tileGen.io.regularTileW := io.regularTileW
  tileGen.io.lastTileW := io.lastTileW
  tileGen.io.tileAccepted := tileAccepted
  io.statusReady := tileGen.io.statusReady

  val sequencer = Module(new F2aSequencer(b, dataAddrWidth, qAddrWidth, dataAddrWidth))
  io.qSync.rawDec := sequencer.io.qRawDec
  sequencer.io.qRawZero := io.qSync.rawZero
  io.aSync.warDec := sequencer.io.aWarDec
  sequencer.io.aWarZero := io.aSync.warZero
  io.fSync.rawDec := sequencer.io.fRawDec
  sequencer.io.fRawZero := io.fSync.rawZero

  sequencer.io.hCount := tileGen.io.tileHeight
  sequencer.io.wCount := tileGen.io.tileWidth
  sequencer.io.tileFirst := tileGen.io.tileFirst
  sequencer.io.tileValid := tileGen.io.tileValid
  tileAccepted := sequencer.io.tileAccepted

  val pipeline = Module(new F2aPipeline(b, fWidth, aWidth, dataAddrWidth, qAddrWidth, dataAddrWidth))
  pipeline.io.control := sequencer.io.control
  io.fmemRead := pipeline.io.fmemRead
  pipeline.io.fmemQ := io.fmemQ
  io.qmemRead := pipeline.io.qmemRead
  pipeline.io.qmemQ := io.qmemQ
  io.amemWrite := pipeline.io.amemWrite

  io.aSync.rawInc := pipeline.io.syncInc.aRaw
  io.qSync.warInc := pipeline.io.syncInc.qWar
  io.fSync.warInc := pipeline.io.syncInc.fWar
}

object F2a {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new F2a(3, 1024, 32, 2, 16)))
  }
}
