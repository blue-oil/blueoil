package bxb.wqdma

import chisel3._
import chisel3.util._

import bxb.memory.{PackedWritePort}
import bxb.util.{Util}

class WQDmaPackedMemoryWriter(avalonDataWidth: Int, wAddrWidth: Int, itemsPerPack: Int, itemWidth: Int, packsPerBlock: Int) extends Module {

  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  require(isPow2(itemsPerPack) && itemsPerPack % avalonDataWidth == 0)
  require(itemsPerPack * itemWidth == avalonDataWidth) // TODO: generalize me

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
    val rawInc = Output(Bool())

    // Memory interface
    val memWrite = Output(PackedWritePort(wAddrWidth, itemsPerPack, itemWidth))
  })

  object State {
    val idle :: running :: Nil = Enum(2)
  }

  val state = RegInit(State.idle)

  val idle = (state === State.idle)
  val running = (state === State.running)

  val acceptData = (running & io.avalonMasterReadDataValid)

  val countLeft = Reg(UInt(Chisel.log2Up(packsPerBlock).W))
  val countLast = (countLeft === 0.U)
  when(idle) {
    countLeft := (packsPerBlock - 1).U
  }.elsewhen(acceptData) {
    countLeft := countLeft - 1.U
  }

  val memAddress = RegInit(0.U(wAddrWidth.W))
  when(acceptData) {
    memAddress := memAddress + 1.U
  }

  val acceptLast = (running & io.avalonMasterReadDataValid & countLast)
  when(idle & io.requesterNext) {
    state := State.running
  }.elsewhen(acceptLast) {
    state := State.idle
  }

  io.memWrite.addr := memAddress
  io.memWrite.data := Seq.tabulate(itemsPerPack){i =>
    val itemMsb = (i + 1) * itemWidth - 1
    val itemLsb = i * itemWidth
    io.avalonMasterReadData(itemMsb, itemLsb)
  }
  io.memWrite.enable := acceptData
  io.writerDone := acceptLast
  io.rawInc := acceptLast
}

object WQDmaPackedMemoryWriter {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new WQDmaPackedMemoryWriter(32, 12, 32, 1, 32)))
  }
}
