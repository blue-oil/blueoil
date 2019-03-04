package bxb.adma

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class ADmaAvalonRequester(avalonAddrWidth: Int, avalonDataWidth: Int, tileCountWidth: Int, maxBurst: Int) extends Module {
  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  val avalonDataByteWidth = avalonDataWidth / 8
  val avalonDataByteWidthLog = Chisel.log2Up(avalonDataByteWidth)
  val burstMsb = Chisel.log2Floor(maxBurst)
  val io = IO(new Bundle {
    // Tile Generator interface
    val tileStartAddress = Input(UInt(avalonAddrWidth.W))
    val tileHeight = Input(UInt(tileCountWidth.W))
    val tileWidth = Input(UInt(tileCountWidth.W))
    // byte distance between last element of some row and
    // first element of a row adjacent to it in elements, could be computed as
    // (inputWidth - tileWidth + (tileWidth % maxBurst == 0) ? maxBurst : tileWidth % maxBurst)
    // let's assume that it is pre computed for now
    val tileRowToRowDistance = Input(UInt(tileCountWidth.W))
    // once tileValid asserted above tile parameters must remain stable and until tileAccepted is asserted
    val tileValid = Input(Bool())
    // accepted at a clock when last request is sent
    val tileAccepted = Output(Bool())

    // AMem writer interface
    val writerDone = Input(Bool())

    // Avalon interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterRead = Output(Bool())
    val avalonMasterBurstCount = Output(UInt(10.W))
    val avalonMasterWaitRequest = Input(Bool())
  })

  private def toBytes(elements: UInt) = {
    elements << avalonDataByteWidthLog.U
  }

  object State {
    val idle :: runningLong :: runningShort :: waitingWriter :: Nil = Enum(4)
  }

  val state = RegInit(State.idle)

  val idle = (state === State.idle)
  val runningLong = (state === State.runningLong)
  val runningShort = (state === State.runningShort)
  val running = (runningLong | runningShort)
  val waitingWriter = (state === State.waitingWriter)

  val waitRequired = (running & io.avalonMasterWaitRequest)

  // tile loops
  val tileXCountShort = if (burstMsb != 0) io.tileWidth(burstMsb - 1, 0) else 0.U(1.W)
  val tileXCountShortZero = (tileXCountShort === 0.U)

  val tileXCountLong = io.tileWidth(tileCountWidth - 1, burstMsb)
  val tileXCountLongZero = (tileXCountLong === 0.U)
  val tileXCountLongLeft = Reg(UInt((tileCountWidth - burstMsb).W))
  val tileXCountLongLast = (tileXCountLongLeft === 1.U)

  val tileXCountLast = (runningShort | (runningLong & tileXCountLongLast & tileXCountShortZero))
  when(~waitRequired) {
    when(idle | tileXCountLongLast) {
      tileXCountLongLeft := tileXCountLong
    }.elsewhen(runningLong) {
      tileXCountLongLeft := tileXCountLongLeft - 1.U
    }
  }

  val tileYCountLeft = Reg(UInt(tileCountWidth.W))
  val tileYCountLast = (tileYCountLeft === 1.U) & tileXCountLast
  when(~waitRequired) {
    when(idle | tileYCountLast) {
      tileYCountLeft := io.tileHeight
    }.elsewhen(tileXCountLast) {
      tileYCountLeft := tileYCountLeft - 1.U
    }
  }

  val avalonAddress = Reg(UInt(avalonAddrWidth.W))
  when(~waitRequired) {
    when(idle) {
      avalonAddress := io.tileStartAddress
    }.elsewhen(tileXCountLast) {
      avalonAddress := avalonAddress + toBytes(io.tileRowToRowDistance)
    }.elsewhen(runningLong) {
      avalonAddress := avalonAddress + toBytes(maxBurst.U)
    }.otherwise {
      avalonAddress := avalonAddress + toBytes(tileXCountShort)
    }
  }

  when(~waitRequired) {
    when(idle & io.tileValid) {
      when(~tileXCountLongZero) {
        state := State.runningLong
      }.elsewhen(~tileXCountShortZero) {
        state := State.runningShort
      }
    }.elsewhen(runningLong & tileXCountLongLast) {
      when(tileYCountLast) {
        state := State.waitingWriter
      }.elsewhen(~tileXCountShortZero) {
        state := State.runningShort
      }
    }.elsewhen(runningShort) {
      when(tileYCountLast) {
        state := State.waitingWriter
      }.elsewhen(~tileXCountLongZero) {
        state := State.runningLong
      }
    }.elsewhen(waitingWriter & io.writerDone) {
      state := State.idle
    }
  }

  io.tileAccepted := waitingWriter
  io.avalonMasterAddress := avalonAddress
  io.avalonMasterRead := running
  io.avalonMasterBurstCount := Mux(runningShort, tileXCountShort, maxBurst.U)
}

object ADmaAvalonRequester {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new ADmaAvalonRequester(32, 64, 12, 4)))
  }
}
