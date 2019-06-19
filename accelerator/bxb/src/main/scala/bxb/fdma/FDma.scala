package bxb.fdma

import chisel3._
import chisel3.util._

import bxb.avalon.{WriteMasterIO}
import bxb.memory.{ReadPort}
import bxb.sync.{ConsumerSyncIO}
import bxb.util.{Util}

class FDma(b: Int, fAddrWidth: Int, avalonAddrWidth: Int, avalonDataWidth: Int, maxBurst: Int) extends Module {
  val fSz = 16
  val tileCountWidth = fAddrWidth
  val dataWidth = b * fSz

  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(isPow2(maxBurst))

  val io = IO(new Bundle {
    val start = Input(Bool())

    // External parameters
    //  - should be provided as stable signals
    val parameters = Input(FDmaParameters(avalonAddrWidth, tileCountWidth))

    // Avalon interface
    val avalonMaster = WriteMasterIO(avalonAddrWidth, avalonDataWidth)

    // FMem interface
    val fmemRead = Output(Vec(b, ReadPort(fAddrWidth)))
    val fmemQ = Input(Vec(b, UInt(fSz.W)))

    // Synchronization interface
    val fSync = ConsumerSyncIO()

    // Status
    val statusReady = Output(Bool())
  })

  val tileAccepted = Wire(Bool())

  val tileGenerator = Module(new FDmaTileGenerator(avalonAddrWidth, dataWidth, tileCountWidth))
  tileGenerator.io.start := io.start
  tileGenerator.io.parameters := io.parameters
  tileGenerator.io.tileAccepted := tileAccepted

  io.fSync.rawDec := tileGenerator.io.fRawDec
  tileGenerator.io.fRawZero := io.fSync.rawZero
  io.fSync.warInc := tileAccepted

  io.statusReady := tileGenerator.io.statusReady

  val fmemReader = Module(new FDmaFMemReader(b, avalonDataWidth, fAddrWidth, tileCountWidth))
  fmemReader.io.tileHeight := tileGenerator.io.tileHeight
  fmemReader.io.tileWidth := tileGenerator.io.tileWidth
  fmemReader.io.tileFirst := tileGenerator.io.tileFirst
  fmemReader.io.tileValid := tileGenerator.io.tileValid
  io.fmemRead := fmemReader.io.fmemRead
  fmemReader.io.fmemQ := io.fmemQ
  io.avalonMaster.writeData := fmemReader.io.data
  fmemReader.io.waitRequest := io.avalonMaster.waitRequest

  val avalonWriter = Module(new FDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, dataWidth, tileCountWidth, maxBurst))
  avalonWriter.io.tileStartAddress := tileGenerator.io.tileStartAddress
  avalonWriter.io.tileHeight := tileGenerator.io.tileHeight
  avalonWriter.io.tileWidth := tileGenerator.io.tileWidth
  avalonWriter.io.tileWordRowToRowDistance := tileGenerator.io.tileWordRowToRowDistance
  avalonWriter.io.tileValid := tileGenerator.io.tileValid
  tileAccepted := avalonWriter.io.tileAccepted
  io.avalonMaster.address := avalonWriter.io.avalonMasterAddress
  io.avalonMaster.burstCount := avalonWriter.io.avalonMasterBurstCount
  avalonWriter.io.avalonMasterWaitRequest := io.avalonMaster.waitRequest
  io.avalonMaster.write := avalonWriter.io.avalonMasterWrite
  avalonWriter.io.readerReady := fmemReader.io.ready
}

object FDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new FDma(32, 10, 16, 128, 4)))
  }
}
