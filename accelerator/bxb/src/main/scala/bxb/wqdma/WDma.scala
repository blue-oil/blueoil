package bxb.wqdma

import chisel3._
import chisel3.util._

import bxb.avalon.{ReadMasterIO}
import bxb.memory.{PackedWritePort}
import bxb.sync.{ProducerSyncIO}
import bxb.util.{Util}

class WDma(b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, wAddrWidth: Int) extends Module {
  require(avalonDataWidth <= b && b % avalonDataWidth == 0)

  // FIXME: rid of copypaste
  val hCountWidth = 6
  val wCountWidth = 6
  val blockCountWidth = 14

  val itemsPerPack = b
  val itemWidth = 1
  val packsPerBlock = b

  val io = IO(new Bundle {
    val start = Input(Bool())

    val startAddress = Input(UInt(avalonAddrWidth.W))
    // - number of output tiles in Height direction
    val outputHCount = Input(UInt(hCountWidth.W))
    // - number of output tiles in Width direction
    val outputWCount = Input(UInt(wCountWidth.W))
    // - (outputC / B * inputC / B * kernelY * kernelX)
    val kernelBlockCount = Input(UInt(blockCountWidth.W))

    // Sync interface
    val wSync = ProducerSyncIO()

    // Avalon interface
    val avalonMaster = ReadMasterIO(avalonAddrWidth, avalonDataWidth)

    // WMem interface
    val wmemWrite = Output(PackedWritePort(wAddrWidth, b, 1))

    // Status
    val statusReady = Output(Bool())
  })

  val writerDone = Wire(Bool())
  val requesterNext = Wire(Bool())

  val requester = Module(new WQDmaAvalonRequester(avalonAddrWidth, avalonDataWidth, itemsPerPack * itemWidth, packsPerBlock))
  requester.io.start := io.start
  requester.io.startAddress := io.startAddress
  requester.io.outputHCount := io.outputHCount
  requester.io.outputWCount := io.outputWCount
  requester.io.blockCount := io.kernelBlockCount

  requesterNext := requester.io.requesterNext
  requester.io.writerDone := writerDone

  requester.io.warZero := io.wSync.warZero
  io.wSync.warDec := requester.io.warDec

  io.avalonMaster.address := requester.io.avalonMasterAddress
  io.avalonMaster.read := requester.io.avalonMasterRead
  io.avalonMaster.burstCount := requester.io.avalonMasterBurstCount
  requester.io.avalonMasterWaitRequest := io.avalonMaster.waitRequest

  io.statusReady := requester.io.statusReady

  val writer = Module(new WQDmaPackedMemoryWriter(avalonDataWidth, wAddrWidth, itemsPerPack, itemWidth, packsPerBlock))
  writer.io.requesterNext := requesterNext
  writerDone := writer.io.writerDone

  writer.io.avalonMasterReadDataValid := io.avalonMaster.readDataValid
  writer.io.avalonMasterReadData := io.avalonMaster.readData

  io.wSync.rawInc := writer.io.rawInc

  io.wmemWrite := writer.io.memWrite
}

object WDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new WDma(32, 32, 32, 12)))
  }
}
