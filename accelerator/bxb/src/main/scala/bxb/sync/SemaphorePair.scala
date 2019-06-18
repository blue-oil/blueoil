package bxb.sync

import chisel3._

import bxb.util.{Util}

class ProducerSyncIO extends Bundle {
  val warDec = Output(Bool())
  val warZero = Input(Bool())
  val rawInc = Output(Bool())
}

object ProducerSyncIO {
  def apply() = {
    new ProducerSyncIO
  }
}

class ConsumerSyncIO extends Bundle {
  val rawDec = Output(Bool())
  val rawZero = Input(Bool())
  val warInc = Output(Bool())
}

object ConsumerSyncIO {
  def apply() = {
    new ConsumerSyncIO
  }
}

class SemaphorePair(counterWidth: Int, initValueRaw: Int, initValueWar: Int) extends Module {
  val io = IO(new Bundle {
    val producer = Flipped(new ProducerSyncIO)
    val consumer = Flipped(new ConsumerSyncIO)
  })

  val warSema = Module(new Semaphore(counterWidth, initValueWar))
  val rawSema = Module(new Semaphore(counterWidth, initValueRaw))

  // producer increments RAW and decrements WAR semaphore
  warSema.io.decrement := io.producer.warDec
  io.producer.warZero := warSema.io.zero
  rawSema.io.increment := io.producer.rawInc
  // consumer increments WAR and decrements RAW semaphore
  rawSema.io.decrement := io.consumer.rawDec
  io.consumer.rawZero := rawSema.io.zero
  warSema.io.increment := io.consumer.warInc
}

class ConsumerSyncMux extends Module {
  val io = IO(new Bundle {
    val select = Input(Bool())
    val out = ConsumerSyncIO()
    val a = Flipped(ConsumerSyncIO())
    val b = Flipped(ConsumerSyncIO())
  })

  io.out.rawDec := Mux(io.select, io.a.rawDec, io.b.rawDec)
  io.a.rawZero := io.out.rawZero
  io.b.rawZero := io.out.rawZero
  io.out.warInc := Mux(io.select, io.a.warInc, io.b.warInc)
}

object SemaphorePair {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new SemaphorePair(2, 0, 3)))
  }
}
