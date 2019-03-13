package bxb.wqdma

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class WQDmaAvalonRequester(avalonAddrWidth: Int, avalonDataWidth: Int, elementWidth: Int, elementsPerBlock: Int) extends Module {

  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  require(isPow2(elementWidth) && elementWidth % avalonDataWidth == 0)

  val hCountWidth = 6
  val wCountWidth = 6
  val blockCountWidth = 14

  val wordsPerElement = elementWidth / avalonDataWidth
  val wordsPerBlock = wordsPerElement * elementsPerBlock
  val bytesPerBlock = elementWidth * elementsPerBlock / 8

  val burstCountWidth = Chisel.log2Ceil(wordsPerBlock) + 1

  val io = IO(new Bundle {
    val start = Input(Bool())

    val startAddress = Input(UInt(avalonAddrWidth.W))
    // - number of output tiles in Height direction
    val outputHCount = Input(UInt(hCountWidth.W))
    // - number of output tiles in Width direction
    val outputWCount = Input(UInt(wCountWidth.W))
    // - number of input blocks to be fetched per tile
    val blockCount = Input(UInt(blockCountWidth.W))

    // WMem writer interface
    val requesterNext = Output(Bool())
    val writerDone = Input(Bool())

    // Sync interface
    val warZero = Input(Bool())
    val warDec = Output(Bool())

    // Avalon interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterRead = Output(Bool())
    val avalonMasterBurstCount = Output(UInt(burstCountWidth.W))
    val avalonMasterWaitRequest = Input(Bool())
  })

  object State {
    val idle :: running :: waitingWriter :: waitWar :: Nil = Enum(4)
  }

  val state = RegInit(State.idle)

  val idle = (state === State.idle)
  val running = (state === State.running)
  val waitingWriter = (state === State.waitingWriter)
  val waitWar = (state === State.waitWar)

  val updateAddress = (idle | (waitingWriter & io.writerDone))

  val blockCountLeft = Reg(UInt(blockCountWidth.W))
  val blockCountLast = (blockCountLeft === 1.U)
  when(updateAddress) {
    when(idle | blockCountLast) {
      blockCountLeft := io.blockCount
    }.otherwise {
      blockCountLeft := blockCountLeft - 1.U
    }
  }

  val outputWCountLeft = Reg(UInt(wCountWidth.W))
  val outputWCountLast = (outputWCountLeft === 1.U) & blockCountLast
  when(updateAddress) {
    when(idle | outputWCountLast) {
      outputWCountLeft := io.outputWCount
    }.elsewhen(blockCountLast) {
      outputWCountLeft := outputWCountLeft - 1.U
    }
  }

  val outputHCountLeft = Reg(UInt(hCountWidth.W))
  val outputHCountLast = (outputHCountLeft === 1.U) & outputWCountLast
  when(updateAddress) {
    when(idle | outputHCountLast) {
      outputHCountLeft := io.outputHCount
    }.elsewhen(outputWCountLast) {
      outputHCountLeft := outputHCountLeft - 1.U
    }
  }

  val avalonAddress = Reg(UInt(avalonAddrWidth.W))
  when(updateAddress) {
    when(idle | blockCountLast) {
      avalonAddress := io.startAddress
    }.otherwise {
      avalonAddress := avalonAddress + bytesPerBlock.U
    }
  }

  when(idle & io.start) {
    state := State.waitWar
  }.elsewhen(waitWar & ~io.warZero) {
    state := State.running
  }.elsewhen(running & ~io.avalonMasterWaitRequest) {
    state := State.waitingWriter
  }.elsewhen(waitingWriter & io.writerDone) {
    when(outputHCountLast) {
      state := State.idle
    }.otherwise {
      state := State.waitWar
    }
  }

  io.warDec := waitWar
  io.requesterNext := running & ~io.avalonMasterWaitRequest

  io.avalonMasterAddress := avalonAddress
  io.avalonMasterRead := running
  // XXX: expect avalon support bursts of suffucient size
  io.avalonMasterBurstCount := wordsPerBlock.U
}

object WQDmaAvalonRequester {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new WQDmaAvalonRequester(32, 32, 32, 32)))
  }
}
