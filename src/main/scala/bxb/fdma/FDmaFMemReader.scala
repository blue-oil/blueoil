package bxb.fdma

import chisel3._
import chisel3.util._

import bxb.memory.{ReadPort}
import bxb.util.{Util, CatLeastFirst}

class FDmaFMemReader(b: Int, avalonDataWidth: Int, fAddrWidth: Int, tileCountWidth: Int) extends Module {
  val fSz = 16
  val rowWidth = fSz * b

  require(isPow2(avalonDataWidth))
  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(avalonDataWidth < rowWidth)

  val featuresPerWord = avalonDataWidth / fSz
  val wordsPerRow = rowWidth / avalonDataWidth

  val io = IO(new Bundle {
    // Tile Generator interface
    val tileHeight = Input(UInt(fAddrWidth.W))
    val tileWidth = Input(UInt(fAddrWidth.W))
    // once tileValid asserted above tile parameters must remain stable and until tileAccepted is asserted
    val tileValid = Input(Bool())

    val fmemRead = Output(Vec(b, ReadPort(fAddrWidth)))
    val fmemQ = Input(Vec(b, UInt(fSz.W)))

    val waitRequest = Input(Bool())
    // reader is ready since first word arrives and untill last word is assigned to data output
    val ready = Output(Bool())

    val data = Output(UInt(avalonDataWidth.W))
  })

  object State {
    val idle :: askFirst :: loadFirst :: running :: acknowledge :: Nil = Enum(5)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val askFirst = (state === State.askFirst)
  val loadFirst = (state === State.loadFirst)
  val running = (state === State.running)
  val acknowledge = (state === State.acknowledge)

  val dataAccepted = (running & ~io.waitRequest)
  val updateCounters = (idle | dataAccepted)

  val wordsCountLeft = Reg(UInt((Chisel.log2Floor(wordsPerRow) + 1).W))
  val wordsCountLast = (wordsCountLeft === 1.U)
  when(updateCounters) {
    when(idle | wordsCountLast) {
      wordsCountLeft := wordsPerRow.U
    }.otherwise {
      wordsCountLeft := wordsCountLeft - 1.U
    }
  }

  // tile loops
  val tileXCountLeft = Reg(UInt(tileCountWidth.W))
  val tileXCountLast = (tileXCountLeft === 1.U) & wordsCountLast
  when(updateCounters) {
    when(idle | tileXCountLast) {
      tileXCountLeft := io.tileWidth
    }.elsewhen(wordsCountLast) {
      tileXCountLeft := tileXCountLeft - 1.U
    }
  }

  val tileYCountLeft = Reg(UInt(tileCountWidth.W))
  val tileYCountLast = (tileYCountLeft === 1.U) & tileXCountLast
  when(updateCounters) {
    when(idle | tileYCountLast) {
      tileYCountLeft := io.tileHeight
    }.elsewhen(tileXCountLast) {
      tileYCountLeft := tileYCountLeft - 1.U
    }
  }

  val fmemBufEvenOdd = RegInit(0.U(1.W))
  when(idle & io.tileValid) {
    fmemBufEvenOdd := ~fmemBufEvenOdd
  }

  val loadData = (loadFirst | (dataAccepted & wordsCountLast))
  val shiftData = dataAccepted

  val fmemAddress = Reg(UInt(fAddrWidth.W))
  when(idle) {
    fmemAddress := Cat(fmemBufEvenOdd, 0.U((fAddrWidth - 1).W))
  }.elsewhen(loadData & ~tileYCountLast) {
    fmemAddress := fmemAddress + 1.U
  }

  for (row <- 0 until b) {
    io.fmemRead(row).addr := fmemAddress
    io.fmemRead(row).enable := (askFirst | running)
  }

  val fmemQPacked = io.fmemQ.grouped(featuresPerWord).map(features => CatLeastFirst(features)).toSeq
  val fmemData = Seq.fill(wordsPerRow)(Reg(UInt(avalonDataWidth.W)))
  for (i <- 0 until wordsPerRow) {
    when(loadData) {
      fmemData(i) := fmemQPacked(i)
    }.elsewhen(shiftData) {
      fmemData(i) := (if (i == wordsPerRow - 1) 0.U else fmemData(i + 1))
    }
  }

  when(idle & io.tileValid) {
    state := State.askFirst
  }.elsewhen(askFirst) {
    state := State.loadFirst
  }.elsewhen(loadFirst) {
    state := State.running
  }.elsewhen(dataAccepted & tileYCountLast) {
    state := State.acknowledge
  }.elsewhen(acknowledge) {
    state := State.idle
  }

  io.ready := (loadFirst | running)
  io.data := fmemData(0)
}

object FDmaFMemReader {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new FDmaFMemReader(32, 128, 12, 12)))
  }
}
