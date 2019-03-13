package bxb.wdma

import chisel3._
import chisel3.util._

import bxb.wqdma.{WQDmaAvalonRequester}
import bxb.memory.{PackedWritePort}
import bxb.util.{Util}

class WDma(b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, wAddrWidth: Int) extends Module {
  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  require(avalonDataWidth == b, "we expect everything to match prefectly")

  // FIXME: rid of copypaste
  val hCountWidth = 6
  val wCountWidth = 6
  val blockCountWidth = 14

  val elementWidth = b
  val elementsPerBlock = b

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
    val wWarZero = Input(Bool())
    val wWarDec = Output(Bool())
    val wRawInc = Output(Bool())

    // Avalon interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterRead = Output(Bool())
    val avalonMasterBurstCount = Output(UInt(10.W))
    val avalonMasterWaitRequest = Input(Bool())

    val avalonMasterReadDataValid = Input(Bool())
    val avalonMasterReadData = Input(UInt(avalonDataWidth.W))

    // WMem interface
    val wmemWrite = Output(PackedWritePort(wAddrWidth, b, 1))
  })

  val writerDone = Wire(Bool())
  val requesterNext = Wire(Bool())

  val requester = Module(new WQDmaAvalonRequester(avalonAddrWidth, avalonDataWidth, elementWidth, elementsPerBlock))
  requester.io.start := io.start
  requester.io.startAddress := io.startAddress
  requester.io.outputHCount := io.outputHCount
  requester.io.outputWCount := io.outputWCount
  requester.io.blockCount := io.kernelBlockCount

  requesterNext := requester.io.requesterNext
  requester.io.writerDone := writerDone

  requester.io.warZero := io.wWarZero
  io.wWarDec := requester.io.warDec

  io.avalonMasterAddress := requester.io.avalonMasterAddress
  io.avalonMasterRead := requester.io.avalonMasterRead
  io.avalonMasterBurstCount := requester.io.avalonMasterBurstCount
  requester.io.avalonMasterWaitRequest := io.avalonMasterWaitRequest

  val writer = Module(new WDmaWMemWriter(b, avalonDataWidth, wAddrWidth))
  writer.io.requesterNext := requesterNext
  writerDone := writer.io.writerDone

  writer.io.avalonMasterReadDataValid := io.avalonMasterReadDataValid
  writer.io.avalonMasterReadData := io.avalonMasterReadData

  io.wRawInc := writer.io.wRawInc

  io.wmemWrite := writer.io.wmemWrite
}

object WDma {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new WDma(32, 32, 32, 12)))
  }
}
