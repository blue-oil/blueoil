package bxb.fdma

import chisel3._
import chisel3.util._

import bxb.memory.{ReadPort}
import bxb.util.{Util}

class FDma(b: Int, fAddrWidth: Int, avalonAddrWidth: Int, avalonDataWidth: Int, maxBurst: Int) extends Module {
  val fSz = 16
  val tileCountWidth = fAddrWidth
  val dataWidth = b * fSz

  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(isPow2(maxBurst))

  val io = IO(new Bundle {
    val start = Input(Bool())

    // Tile generation parameters
    val outputAddress = Input(UInt(avalonAddrWidth.W))
    // - should be equal to roundUp(outputHeight / tileHeight)
    val outputHCount = Input(UInt(6.W))
    // - should be equal to roundUp(outputWidth / tileWidth)
    val outputWCount = Input(UInt(6.W))
    // - should be equal to roundUp(outputChannels / B)
    val outputCCount = Input(UInt(6.W))

    // tileHeight
    val regularTileH = Input(UInt(tileCountWidth.W))
    // - outputHeight - (hCount - 1)  * tileHeight
    val lastTileH = Input(UInt(tileCountWidth.W))

    // tileWidth
    val regularTileW = Input(UInt(tileCountWidth.W))
    // - outputWidth - (wCount - 1)  * tileWidth
    val lastTileW = Input(UInt(tileCountWidth.W))

    // (outputWidth - regularTileW + (regularTileW % maxBurst == 0) ? maxBurst : regularTileW % maxBurst)
    val regularRowToRowDistance = Input(UInt(tileCountWidth.W))
    // (outputWidth - lastTileW + (lastTileW % maxBurst == 0) ? maxBurst : lastTileW % maxBurst)
    val lastRowToRowDistance = Input(UInt(tileCountWidth.W))

    // outputHeight * outputWidth
    val outputSpace = Input(UInt(avalonAddrWidth.W))

    // outputWidth * regularTileH - outputWidth + lastTileW
    val rowDistance = Input(UInt(avalonAddrWidth.W))

    // Avalon interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterBurstCount = Output(UInt(10.W))
    val avalonMasterWaitRequest = Input(Bool())
    val avalonMasterWrite = Output(Bool())
    val avalonMasterWriteData = Output(UInt(avalonDataWidth.W))

    // FMem interface
    val fmemRead = Output(Vec(b, ReadPort(fAddrWidth)))
    val fmemQ = Input(Vec(b, UInt(fSz.W)))

    // Synchronization interface
    val fRawDec = Output(Bool())
    val fRawZero = Input(Bool())
    val fWarInc = Output(Bool())

    // Status
    val statusReady = Output(Bool())
  })

  val tileAccepted = Wire(Bool())

  val tileGenerator = Module(new FDmaTileGenerator(avalonAddrWidth, dataWidth, tileCountWidth))
  tileGenerator.io.start := io.start

  tileGenerator.io.outputAddress := io.outputAddress
  tileGenerator.io.outputHCount := io.outputHCount
  tileGenerator.io.outputWCount := io.outputWCount
  tileGenerator.io.outputCCount := io.outputCCount

  tileGenerator.io.regularTileH := io.regularTileH
  tileGenerator.io.lastTileH := io.lastTileH

  tileGenerator.io.regularTileW := io.regularTileW
  tileGenerator.io.lastTileW := io.lastTileW

  tileGenerator.io.regularRowToRowDistance := io.regularRowToRowDistance
  tileGenerator.io.lastRowToRowDistance := io.lastRowToRowDistance

  tileGenerator.io.outputSpace := io.outputSpace
  tileGenerator.io.rowDistance := io.rowDistance

  tileGenerator.io.tileAccepted := tileAccepted

  io.fRawDec := tileGenerator.io.fRawDec
  tileGenerator.io.fRawZero := io.fRawZero
  io.fWarInc := tileAccepted

  io.statusReady := tileGenerator.io.statusReady

  val fmemReader = Module(new FDmaFMemReader(b, avalonDataWidth, fAddrWidth, tileCountWidth))
  fmemReader.io.tileHeight := tileGenerator.io.tileHeight
  fmemReader.io.tileWidth := tileGenerator.io.tileWidth
  fmemReader.io.tileFirst := tileGenerator.io.tileFirst
  fmemReader.io.tileValid := tileGenerator.io.tileValid
  io.fmemRead := fmemReader.io.fmemRead
  fmemReader.io.fmemQ := io.fmemQ
  io.avalonMasterWriteData := fmemReader.io.data
  fmemReader.io.waitRequest := io.avalonMasterWaitRequest

  val avalonWriter = Module(new FDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, dataWidth, tileCountWidth, maxBurst))
  avalonWriter.io.tileStartAddress := tileGenerator.io.tileStartAddress
  avalonWriter.io.tileHeight := tileGenerator.io.tileHeight
  avalonWriter.io.tileWidth := tileGenerator.io.tileWidth
  avalonWriter.io.tileWordRowToRowDistance := tileGenerator.io.tileWordRowToRowDistance
  avalonWriter.io.tileValid := tileGenerator.io.tileValid
  tileAccepted := avalonWriter.io.tileAccepted
  io.avalonMasterAddress := avalonWriter.io.avalonMasterAddress
  io.avalonMasterBurstCount := avalonWriter.io.avalonMasterBurstCount
  avalonWriter.io.avalonMasterWaitRequest := io.avalonMasterWaitRequest
  io.avalonMasterWrite := avalonWriter.io.avalonMasterWrite
  avalonWriter.io.readerReady := fmemReader.io.ready
}

object FDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new FDma(32, 10, 16, 128, 4)))
  }
}
