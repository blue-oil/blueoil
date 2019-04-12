package bxb.f2a

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.sync.{ConsumerSyncIO, ProducerSyncIO}

import bxb.util.{Util}

class F2a(b: Int, dataMemSize: Int, qmemSize: Int, aWidth: Int, fWidth: Int, qWidth: Int) extends Module {
  val dataAddrWidth = Chisel.log2Up(dataMemSize)
  val qAddrWidth = Chisel.log2Up(qmemSize)
  val io = IO(new Bundle {

    val hCount = Input(UInt(dataAddrWidth.W))
    val wCount = Input(UInt(dataAddrWidth.W))

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

    // TODO: Is the statusReady need to CSR?
    // Status
    //val statusReady = Output(Bool())
  })

  val sequencer = Module(new F2aSequencer(b, fWidth, qWidth, aWidth, dataAddrWidth, qAddrWidth, dataAddrWidth))
  io.qSync.rawDec := sequencer.io.qRawDec
  sequencer.io.qRawZero := io.qSync.rawZero
  io.aSync.warDec := sequencer.io.aWarDec
  sequencer.io.aWarZero := io.aSync.warZero
  io.fSync.rawDec := sequencer.io.fRawDec
  sequencer.io.fRawZero := io.fSync.rawZero

  sequencer.io.hCount := io.hCount
  sequencer.io.wCount := io.wCount

  io.fmemRead := sequencer.io.fmemRead
  io.qmemRead := sequencer.io.qmemRead

  val pipeline = Module(new F2aPipeline(b, fWidth, qWidth, aWidth, dataAddrWidth))
  pipeline.io.control := sequencer.io.control
  pipeline.io.fMemQ := io.fmemQ
  pipeline.io.qMemQ := io.qmemQ
  pipeline.io.amemWriteAddr := sequencer.io.amemWriteAddr
  pipeline.io.writeEnable := sequencer.io.writeEnable

  io.amemWrite := pipeline.io.amemWrite
}

object F2a {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new F2a(3, 1024, 32, 2, 16, 13)))
  }
}
