package bxb.rdma

import chisel3._
import chisel3.util._

import bxb.memory.{ReadPort}
import bxb.util.{Util}

class RDma(b: Int, rAddrWidth: Int, avalonAddrWidth: Int, maxBurst: Int) extends Module {
  val rSz = 2
  val tileCountWidth = rAddrWidth

  val avalonDataWidth = b * rSz
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
    val outputSpace = Input(UInt(tileCountWidth.W))

    // outputWidth * regularTileH - outputWidth + lastTileW
    val rowDistance = Input(UInt(avalonAddrWidth.W))

    // Avalon interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterBurstCount = Output(UInt(10.W))
    val avalonMasterWaitRequest = Input(Bool())
    val avalonMasterWrite = Output(Bool())
    val avalonMasterWriteData = Output(UInt(avalonDataWidth.W))

    // RMem interface
    val rmemRead = Output(Vec(b, ReadPort(rAddrWidth)))
    val rmemQ = Input(Vec(b, UInt(rSz.W)))

    // Synchronization interface
    val rRawDec = Output(Bool())
    val rRawZero = Input(Bool())
    val rWarInc = Output(Bool())

    // Status
    val statusReady = Output(Bool())
  })

  val tileAccepted = Wire(Bool())

  val tileGenerator = Module(new RDmaTileGenerator(avalonAddrWidth, avalonDataWidth, tileCountWidth))
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

  io.rRawDec := tileGenerator.io.rRawDec
  tileGenerator.io.rRawZero := io.rRawZero
  io.rWarInc := tileAccepted

  io.statusReady := tileGenerator.io.statusReady

  val rmemReader = Module(new RDmaBufferedRMemReader(b, avalonDataWidth, rAddrWidth, tileCountWidth))
  rmemReader.io.tileHeight := tileGenerator.io.tileHeight
  rmemReader.io.tileWidth := tileGenerator.io.tileWidth
  rmemReader.io.tileValid := tileGenerator.io.tileValid
  io.rmemRead := rmemReader.io.rmemRead
  rmemReader.io.rmemQ := io.rmemQ
  io.avalonMasterWriteData := rmemReader.io.data
  rmemReader.io.waitRequest := io.avalonMasterWaitRequest

  val avalonWriter = Module(new RDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))
  avalonWriter.io.tileStartAddress := tileGenerator.io.tileStartAddress
  avalonWriter.io.tileHeight := tileGenerator.io.tileHeight
  avalonWriter.io.tileWidth := tileGenerator.io.tileWidth
  avalonWriter.io.tileRowToRowDistance := tileGenerator.io.tileRowToRowDistance
  avalonWriter.io.tileValid := tileGenerator.io.tileValid
  tileAccepted := avalonWriter.io.tileAccepted
  io.avalonMasterAddress := avalonWriter.io.avalonMasterAddress
  io.avalonMasterBurstCount := avalonWriter.io.avalonMasterBurstCount
  avalonWriter.io.avalonMasterWaitRequest := io.avalonMasterWaitRequest
  io.avalonMasterWrite := avalonWriter.io.avalonMasterWrite
  avalonWriter.io.readerReady := rmemReader.io.ready
}

object RDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new RDma(32, 10, 16, 4)))
  }
}
