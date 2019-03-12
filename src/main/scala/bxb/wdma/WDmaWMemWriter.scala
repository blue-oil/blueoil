package bxb.wdma

import chisel3._
import chisel3.util._

import bxb.memory.{PackedWritePort}
import bxb.util.{Util}

class WDmaWMemWriter(b: Int, avalonDataWidth: Int, wAddrWidth: Int) extends Module {
  require(avalonDataWidth == b, "we expect everything to match prefectly")
  val io = IO(new Bundle {
    // Avalon Requester interface
    // one cycle pulse after request was successfully sent
    val requesterNext = Input(Bool())
    // one cycle pulse after all data were accepted
    val writerDone = Output(Bool())

    // Avalon Interface
    val avalonMasterReadDataValid = Input(Bool())
    val avalonMasterReadData = Input(UInt(avalonDataWidth.W))

    // Sync interface
    val wRawInc = Output(Bool())

    // WMem interface
    val wmemWrite = Output(PackedWritePort(wAddrWidth, b, 1))
  })

  object State {
    val idle :: running :: Nil = Enum(2)
  }

  val state = RegInit(State.idle)

  val idle = (state === State.idle)
  val running = (state === State.running)

  val acceptData = (running & io.avalonMasterReadDataValid)

  val countLeft = Reg(UInt(Chisel.log2Up(b).W))
  val countLast = (countLeft === 0.U)
  when(idle) {
    countLeft := (b - 1).U
  }.elsewhen(acceptData) {
    countLeft := countLeft - 1.U
  }

  val wmemAddress = RegInit(0.U(wAddrWidth.W))
  when(acceptData) {
    wmemAddress := wmemAddress + 1.U
  }

  val acceptLast = (running & io.avalonMasterReadDataValid & countLast)
  when(idle & io.requesterNext) {
    state := State.running
  }.elsewhen(acceptLast) {
    state := State.idle
  }

  io.wmemWrite.addr := wmemAddress
  io.wmemWrite.data := Seq.tabulate(b){i => io.avalonMasterReadData(i)}
  io.wmemWrite.enable := acceptData
  io.writerDone := acceptLast
  io.wRawInc := acceptLast
}

object WDmaWMemWriter {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new WDmaWMemWriter(32, 32, 12)))
  }
}
