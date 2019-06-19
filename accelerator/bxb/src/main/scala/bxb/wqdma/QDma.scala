package bxb.wqdma

import chisel3._
import chisel3.util._

import bxb.avalon.{ReadMasterIO}
import bxb.memory.{PackedWritePort}
import bxb.sync.{ProducerSyncIO}
import bxb.util.{Util}

class QDma(b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, wAddrWidth: Int) extends Module {
  val thresholdWidth = 13
  val thresholdWidthRaw = 16
  // one thresholds triple contains 3 thresholds and one sign
  val thresholdsTriplesPerPack = b
  val thresholdsTripleWidth = thresholdWidth * 3 + 1
  val thresholdsTripleWidthRaw = thresholdWidthRaw * 4
  val packsPerBlock = 1

  val requester = Module(new WQDmaAvalonRequester(avalonAddrWidth, avalonDataWidth, thresholdsTriplesPerPack * thresholdsTripleWidthRaw, packsPerBlock))

  val io = IO(new Bundle {
    val start = Input(Bool())

    val parameters = Input(WQDmaParameters(avalonAddrWidth))

    // Sync interface
    val qSync = ProducerSyncIO()

    // Avalon interface
    val avalonMaster = ReadMasterIO(avalonAddrWidth, avalonDataWidth)

    // QMem interface
    val qmemWrite = Output(PackedWritePort(wAddrWidth, b, thresholdsTripleWidth))

    // Status
    val statusReady = Output(Bool())
  })

  val writerDone = Wire(Bool())
  val requesterNext = Wire(Bool())

  requester.io.start := io.start
  requester.io.parameters := io.parameters

  requesterNext := requester.io.requesterNext
  requester.io.writerDone := writerDone

  requester.io.warZero := io.qSync.warZero
  io.qSync.warDec := requester.io.warDec

  io.avalonMaster.address := requester.io.avalonMasterAddress
  io.avalonMaster.read := requester.io.avalonMasterRead
  io.avalonMaster.burstCount := requester.io.avalonMasterBurstCount
  requester.io.avalonMasterWaitRequest := io.avalonMaster.waitRequest

  io.statusReady := requester.io.statusReady

  class QDmaPackedMemoryWriter extends WQDmaPackedMemoryWriter(avalonDataWidth, wAddrWidth, thresholdsTriplesPerPack, thresholdsTripleWidth, packsPerBlock, thresholdsTripleWidthRaw) {
    // XXX: we may need to generalize it later,
    // not the assumption simplifies unpackRead implementation
    require(avalonDataWidth == thresholdsTripleWidthRaw)

    override def unpackRead(packed: UInt) = {
      val chunks = (0 until thresholdsTripleWidthRaw by thresholdWidthRaw).map(lsb => packed(lsb + thresholdWidthRaw - 1, lsb)).toSeq
      val th = (0 until 3).map(i => chunks(i)(thresholdWidth - 1, 0))
      val sign = chunks(3)(thresholdWidthRaw - 1)
      Cat(sign, th(2), th(1), th(0))
    }
  }

  val writer = Module(new QDmaPackedMemoryWriter)
  writer.io.requesterNext := requesterNext
  writerDone := writer.io.writerDone

  writer.io.avalonMasterReadDataValid := io.avalonMaster.readDataValid
  writer.io.avalonMasterReadData := io.avalonMaster.readData

  io.qSync.rawInc := writer.io.rawInc

  io.qmemWrite := writer.io.memWrite
}

object QDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new QDma(32, 32, 64, 12)))
  }
}
