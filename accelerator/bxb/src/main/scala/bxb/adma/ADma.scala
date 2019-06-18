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

    // Input geometry parameters - begin
    // Tile generation parameters
    val inputAddress = Input(UInt(avalonAddrWidth.W))
    // - should be equal to roundUp(inputHeight / (tileHeight - pad))
    val inputHCount = Input(UInt(6.W))
    // - should be equal to roundUp(inputWidth / (tileWidth - pad))
    val inputWCount = Input(UInt(6.W))
    // - should be equal to roundUp(inputChannels / B)
    val inputCCount = Input(UInt(6.W))
    // - should be equal to roundUp(outputChannels / B)
    val outputCCount = Input(UInt(6.W))

    // - tileHeight - pad
    val topTileH = Input(UInt(tileCountWidth.W))
    // - tileHeight
    val middleTileH = Input(UInt(tileCountWidth.W))
    // - inputHeight + pad - (hCount - 1)  * (tileHeight - pad)
    val bottomTileH = Input(UInt(tileCountWidth.W))

    // - tileWidth - pad
    val leftTileW = Input(UInt(tileCountWidth.W))
    // - tileWidth
    val middleTileW = Input(UInt(tileCountWidth.W))
    // - inputWidth + pad - (wCount - 1) * (tileWidth - pad)
    val rightTileW = Input(UInt(tileCountWidth.W))

    // (inputWidth - leftTileW + (leftTileW % maxBurst == 0) ? maxBurst : leftTileW % maxBurst)
    val leftRowToRowDistance = Input(UInt(tileCountWidth.W))
    // (inputWidth - middleTileW + (middleTileW % maxBurst == 0) ? maxBurst : middleTileW % maxBurst)
    val middleRowToRowDistance = Input(UInt(tileCountWidth.W))
    // (inputWidth - rightTileW + (rightTileW % maxBurst == 0) ? maxBurst : rightTileW % maxBurst)
    val rightRowToRowDistance = Input(UInt(tileCountWidth.W))

    // leftTileW - pad
    val leftStep = Input(UInt(avalonAddrWidth.W))
    // middleTileW - pad
    val middleStep = Input(UInt(avalonAddrWidth.W))

    // inputWidth * (topTileH - pad) - inputWidth + rightTileW
    val topRowDistance = Input(UInt(avalonAddrWidth.W))
    // inputWidth * (middleTileH - pad) - inputWidth + rightTileW
    val midRowDistance = Input(UInt(avalonAddrWidth.W))

    // inputWidth * inputHeight
    val inputSpace = Input(UInt(avalonAddrWidth.W))

    // (leftTileW + pad) * pad
    val topBottomLeftPad = Input(UInt(tileCountWidth.W))
    // middleTileW * pad
    val topBottomMiddlePad = Input(UInt(tileCountWidth.W))
    // (rightTileW + pad) * pad
    val topBottomRightPad = Input(UInt(tileCountWidth.W))
    // pad
    val sidePad = Input(UInt(tileCountWidth.W))
    // Input geometry parameters --- end

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
  tileGenerator.io.inputAddress := io.inputAddress
  tileGenerator.io.inputHCount := io.inputHCount
  tileGenerator.io.inputWCount := io.inputWCount
  tileGenerator.io.inputCCount := io.inputCCount
  tileGenerator.io.outputCCount := io.outputCCount
  tileGenerator.io.topTileH := io.topTileH
  tileGenerator.io.middleTileH := io.middleTileH
  tileGenerator.io.bottomTileH := io.bottomTileH
  tileGenerator.io.leftTileW := io.leftTileW
  tileGenerator.io.middleTileW := io.middleTileW
  tileGenerator.io.rightTileW := io.rightTileW
  tileGenerator.io.leftRowToRowDistance := io.leftRowToRowDistance
  tileGenerator.io.middleRowToRowDistance := io.middleRowToRowDistance
  tileGenerator.io.rightRowToRowDistance := io.rightRowToRowDistance
  tileGenerator.io.leftStep := io.leftStep
  tileGenerator.io.middleStep := io.middleStep
  tileGenerator.io.topRowDistance := io.topRowDistance
  tileGenerator.io.midRowDistance := io.midRowDistance
  tileGenerator.io.inputSpace := io.inputSpace
  tileGenerator.io.topBottomLeftPad := io.topBottomLeftPad
  tileGenerator.io.topBottomMiddlePad := io.topBottomMiddlePad
  tileGenerator.io.topBottomRightPad := io.topBottomRightPad
  tileGenerator.io.sidePad := io.sidePad
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
