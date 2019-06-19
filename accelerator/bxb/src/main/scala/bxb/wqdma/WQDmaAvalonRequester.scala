package bxb.wqdma

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class WQDmaParameters(private val avalonAddrWidth: Int) extends Bundle {
  private val hCountWidth = 6
  private val wCountWidth = 6
  private val blockCountWidth = 14

  val startAddress = UInt(avalonAddrWidth.W)
  // - number of output tiles in Height direction
  val outputHCount = UInt(hCountWidth.W)
  // - number of output tiles in Width direction
  val outputWCount = UInt(wCountWidth.W)
  // - (outputC / B * inputC / B * kernelY * kernelX)
  val blockCount = UInt(blockCountWidth.W)
}

object WQDmaParameters {
  def apply(avalonAddrWidth: Int) = {
    new WQDmaParameters(avalonAddrWidth)
  }
}

class WQDmaAvalonRequester(avalonAddrWidth: Int, avalonDataWidth: Int, elementWidth: Int, elementsPerBlock: Int) extends Module {

  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  require(isPow2(elementWidth) && elementWidth % avalonDataWidth == 0)

  val wCountWidth = 6
  val blockCountWidth = 14

  val wordsPerElement = elementWidth / avalonDataWidth
  val wordsPerBlock = wordsPerElement * elementsPerBlock
  val bytesPerBlock = elementWidth * elementsPerBlock / 8

  val burstCountWidth = Chisel.log2Ceil(wordsPerBlock) + 1

  val io = IO(new Bundle {
    val start = Input(Bool())

    val parameters = Input(WQDmaParameters(avalonAddrWidth))

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

    // Status
    val statusReady = Output(Bool())
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

  val blockCountLeft = Reg(io.parameters.blockCount.cloneType)
  val blockCountLast = (blockCountLeft === 1.U)
  when(updateAddress) {
    when(idle | blockCountLast) {
      blockCountLeft := io.parameters.blockCount
    }.otherwise {
      blockCountLeft := blockCountLeft - 1.U
    }
  }

  val outputWCountLeft = Reg(io.parameters.outputWCount.cloneType)
  val outputWCountLast = (outputWCountLeft === 1.U) & blockCountLast
  when(updateAddress) {
    when(idle | outputWCountLast) {
      outputWCountLeft := io.parameters.outputWCount
    }.elsewhen(blockCountLast) {
      outputWCountLeft := outputWCountLeft - 1.U
    }
  }

  val outputHCountLeft = Reg(io.parameters.outputHCount.cloneType)
  val outputHCountLast = (outputHCountLeft === 1.U) & outputWCountLast
  when(updateAddress) {
    when(idle | outputHCountLast) {
      outputHCountLeft := io.parameters.outputHCount
    }.elsewhen(outputWCountLast) {
      outputHCountLeft := outputHCountLeft - 1.U
    }
  }

  val avalonAddress = Reg(UInt(avalonAddrWidth.W))
  when(updateAddress) {
    when(idle | blockCountLast) {
      avalonAddress := io.parameters.startAddress
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
  io.statusReady := idle
}

object WQDmaAvalonRequester {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new WQDmaAvalonRequester(32, 32, 32, 32)))
  }
}
