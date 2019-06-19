package bxb.rdma

import chisel3._
import chisel3.util._

import bxb.avalon.{WriteMasterIO}
import bxb.memory.{ReadPort}
import bxb.sync.{ConsumerSyncIO}
import bxb.util.{Util}

class RDma(b: Int, rAddrWidth: Int, avalonAddrWidth: Int, maxBurst: Int) extends Module {
  val rSz = 2
  val tileCountWidth = rAddrWidth

  val avalonDataWidth = b * rSz
  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(isPow2(maxBurst))

  val io = IO(new Bundle {
    val start = Input(Bool())

    val parameters = Input(RDmaParameters(avalonAddrWidth, tileCountWidth))

    // Avalon interface
    val avalonMaster = WriteMasterIO(avalonAddrWidth, avalonDataWidth)

    // RMem interface
    val rmemRead = Output(Vec(b, ReadPort(rAddrWidth)))
    val rmemQ = Input(Vec(b, UInt(rSz.W)))

    // Synchronization interface
    val rSync = ConsumerSyncIO()

    // Status
    val statusReady = Output(Bool())
  })

  val tileAccepted = Wire(Bool())

  val tileGenerator = Module(new RDmaTileGenerator(avalonAddrWidth, avalonDataWidth, tileCountWidth))
  tileGenerator.io.start := io.start
  tileGenerator.io.parameters := io.parameters
  tileGenerator.io.tileAccepted := tileAccepted

  io.rSync.rawDec := tileGenerator.io.rRawDec
  tileGenerator.io.rRawZero := io.rSync.rawZero
  io.rSync.warInc := tileAccepted

  io.statusReady := tileGenerator.io.statusReady

  val rmemReader = Module(new RDmaBufferedRMemReader(b, avalonDataWidth, rAddrWidth, tileCountWidth))
  rmemReader.io.tileHeight := tileGenerator.io.tileHeight
  rmemReader.io.tileWidth := tileGenerator.io.tileWidth
  rmemReader.io.tileFirst := tileGenerator.io.tileFirst
  rmemReader.io.tileValid := tileGenerator.io.tileValid
  io.rmemRead := rmemReader.io.rmemRead
  rmemReader.io.rmemQ := io.rmemQ
  io.avalonMaster.writeData := rmemReader.io.data
  rmemReader.io.waitRequest := io.avalonMaster.waitRequest

  val avalonWriter = Module(new RDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))
  avalonWriter.io.tileStartAddress := tileGenerator.io.tileStartAddress
  avalonWriter.io.tileHeight := tileGenerator.io.tileHeight
  avalonWriter.io.tileWidth := tileGenerator.io.tileWidth
  avalonWriter.io.tileRowToRowDistance := tileGenerator.io.tileRowToRowDistance
  avalonWriter.io.tileValid := tileGenerator.io.tileValid
  tileAccepted := avalonWriter.io.tileAccepted
  io.avalonMaster.address := avalonWriter.io.avalonMasterAddress
  io.avalonMaster.burstCount := avalonWriter.io.avalonMasterBurstCount
  avalonWriter.io.avalonMasterWaitRequest := io.avalonMaster.waitRequest
  io.avalonMaster.write := avalonWriter.io.avalonMasterWrite
  avalonWriter.io.readerReady := rmemReader.io.ready
}

object RDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new RDma(32, 10, 16, 4)))
  }
}
