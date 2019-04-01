package bxb.w2m

import chisel3._

import bxb.memory.{ReadPort}
import bxb.sync.{ProducerSyncIO, ConsumerSyncIO}

import bxb.util.{Util}

class W2m(b: Int, wmemSize: Int) extends Module {
  val wAddrWidth = Chisel.log2Up(wmemSize)
  val io = IO(new Bundle {
    // WMem interface
    val wmemRead = Output(ReadPort(wAddrWidth))
    val wmemQ = Input(Vec(b, UInt(1.W)))
    // Mac Array interface
    val mOut = Output(Vec(b, Vec(2, UInt(1.W))))
    val mWe = Output(Vec(2, Bool()))
    // Sync interface
    val wSync = ConsumerSyncIO()
    val mSync = ProducerSyncIO()
  })

  val sequencer = Module(new W2mSequencer(b, wAddrWidth))
  sequencer.io.mWarZero := io.mSync.warZero
  io.mSync.warDec := sequencer.io.mWarDec
  sequencer.io.wRawZero := io.wSync.rawZero
  io.wSync.rawDec := sequencer.io.wRawDec

  val pipeline = Module(new W2mPipeline(b, wAddrWidth))
  pipeline.io.control := sequencer.io.control
  io.wmemRead := pipeline.io.wmemRead
  pipeline.io.wmemQ := io.wmemQ

  io.mOut := pipeline.io.mOut
  io.mWe := pipeline.io.mWe

  io.wSync.warInc := pipeline.io.wWarInc
  io.mSync.rawInc := pipeline.io.mRawInc
}

object W2m {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new W2m(3, 1024)))
  }
}
