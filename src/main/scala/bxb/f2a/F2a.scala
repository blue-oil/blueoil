package bxb.f2a

import chisel3._

import bxb.memory.{ReadPort, WritePort}
import bxb.sync.{ConsumerSyncIO, ProducerSyncIO}

import bxb.util.{Util}

class F2a(b: Int, memSize: Int, aWidth: Int, fWidth: Int, qWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(memSize)
  val io = IO(new Bundle {

    val hCount = Input(UInt(addrWidth.W))
    val wCount = Input(UInt(addrWidth.W))

    // AMem interface
    val amemRead = Output(Vec(b, ReadPort(addrWidth)))
    val amemWrite = Output(Vec(b, WritePort(addrWidth, aWidth)))
    val amemQ = Input(Vec(b, UInt(aWidth.W)))

    // FMem interface
    val fmemRead = Output(Vec(b, ReadPort(addrWidth)))
    val fmemWrite = Output(Vec(b, WritePort(addrWidth, fWidth)))
    val fmemQ = Input(Vec(b, UInt(fWidth.W)))

    // QMem interface
    val qmemRead = Output(ReadPort(addrWidth))
    val qmemQ = Input(Vec(b, UInt(1.W)))

    // Sync interface
    val aSync = ProducerSyncIO()
    val qSync = ConsumerSyncIO()
    val fSync = ConsumerSyncIO()

    // TODO: Is the statusReady need to CSR?
    // Status
    //val statusReady = Output(Bool())
  })

  val sequencer = Module(new F2aSequencer(b, fWidth, qWidth, aWidth, addrWidth, addrWidth, addrWidth))
  sequencer.io.control := pipeline.io.control

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
  io.amemWrite := sequencer.io.amemWriteAddr

  val pipeline = Module(new F2aPipeline(b, fWidth, qWidth, aWidth, addrWidth))
  pipeline.io.control := sequencer.io.control
  pipeline.io.fMemQ := io.fmemQ
  pipeline.io.qMemQ := io.qmemQ
  pipeline.io.amemWriteAddr := io.amemRead
  pipeline.io.writeEnable := sequencer.io.writeEnable

  io.amemWrite := pipeline.io.amemWrite
  pipeline.io.writeEnable := sequencer.io.writeEnable

}

object F2a {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new F2a(3, 1024, 2, 16, 13)))
  }
}
