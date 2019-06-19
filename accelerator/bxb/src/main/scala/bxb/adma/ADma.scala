package bxb.adma

import chisel3._
import chisel3.util._

import bxb.avalon.{ReadMasterIO}
import bxb.memory.{WritePort}
import bxb.sync.{ProducerSyncIO}
import bxb.util.{Util}

class ADma(b: Int, aAddrWidth: Int, avalonAddrWidth: Int, maxBurst: Int) extends Module {
  val aSz = 2
  val tileCountWidth = aAddrWidth

  // we could assume that each transaction bring us "one pixel" data for now
  val avalonDataWidth = b * aSz
  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(isPow2(maxBurst))

  val io = IO(new Bundle {
    val start = Input(Bool())

    // External parameters
    // - should be provided as stable signals
    val parameters = Input(ADmaParameters(avalonAddrWidth, tileCountWidth))

    val avalonMaster = ReadMasterIO(avalonAddrWidth, avalonDataWidth)

    val amemWrite = Output(Vec(b, WritePort(aAddrWidth, aSz)))

    val aSync = ProducerSyncIO()

    // Status
    val statusReady = Output(Bool())
  })

  val tileAcceptedByRequester = Wire(Bool())
  val tileAcceptedByWriter = Wire(Bool())
  val tileDone = tileAcceptedByRequester & tileAcceptedByWriter

  io.aSync.rawInc := tileDone

  val tileGenerator = Module(new ADmaTileGenerator(avalonAddrWidth, avalonDataWidth, tileCountWidth))
  tileGenerator.io.start := io.start
  tileGenerator.io.parameters := io.parameters
  tileGenerator.io.tileAccepted := tileDone
  tileGenerator.io.aWarZero := io.aSync.warZero
  io.aSync.warDec := tileGenerator.io.aWarDec
  io.statusReady := tileGenerator.io.statusReady

  // Destination Address Generator
  val amemWriter = Module(new ADmaAMemWriter(b, avalonDataWidth, aAddrWidth, tileCountWidth))
  amemWriter.io.tileHeight := tileGenerator.io.tileHeight
  amemWriter.io.tileWidth := tileGenerator.io.tileWidth
  amemWriter.io.tileStartPad := tileGenerator.io.tileStartPad
  amemWriter.io.tileSidePad := tileGenerator.io.tileSidePad
  amemWriter.io.tileEndPad := tileGenerator.io.tileEndPad
  amemWriter.io.tileValid := tileGenerator.io.tileValid
  amemWriter.io.tileFirst := tileGenerator.io.tileFirst
  tileAcceptedByWriter := amemWriter.io.tileAccepted
  amemWriter.io.avalonMasterReadDataValid := io.avalonMaster.readDataValid
  amemWriter.io.avalonMasterReadData := io.avalonMaster.readData
  io.amemWrite := amemWriter.io.amemWrite

  // Read Request Generator
  val avalonRequester = Module(new ADmaAvalonRequester(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))
  avalonRequester.io.tileStartAddress := tileGenerator.io.tileStartAddress
  avalonRequester.io.tileHeight := tileGenerator.io.tileHeight
  avalonRequester.io.tileWidth := tileGenerator.io.tileWidth
  avalonRequester.io.tileRowToRowDistance := tileGenerator.io.tileRowToRowDistance
  avalonRequester.io.tileValid := tileGenerator.io.tileValid
  tileAcceptedByRequester := avalonRequester.io.tileAccepted
  avalonRequester.io.writerDone := amemWriter.io.writerDone
  io.avalonMaster.address := avalonRequester.io.avalonMasterAddress
  io.avalonMaster.read := avalonRequester.io.avalonMasterRead
  io.avalonMaster.burstCount := avalonRequester.io.avalonMasterBurstCount
  avalonRequester.io.avalonMasterWaitRequest := io.avalonMaster.waitRequest
}

object ADma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new ADma(32, 10, 16, 4)))
  }
}
