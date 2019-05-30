package bxb.wqdma

import chisel3._
import chisel3.util._

import bxb.memory.{PackedWritePort}
import bxb.util.{Util}

// like ShiftRegiter but provides parallel access to all stored elements
private object ShiftingBuffer {
  def apply(depth: Int, in: UInt, enable: Bool): UInt = {
    if (depth == 0) {
      in
    }
    else {
      val shifter = Seq.fill(depth){Reg(in.cloneType)}
      for (i <- 0 until depth) {
        when(enable) {
          shifter(i) := (if (i == 0) in else shifter(i - 1))
        }
      }
      Cat(shifter)
    }
  }
}

class WQDmaPackedMemoryWriter(avalonDataWidth: Int, wAddrWidth: Int, itemsPerPack: Int, itemWidth: Int, packsPerBlock: Int, itemWidthRaw: Int) extends Module {

  def this(avalonDataWidth: Int, wAddrWidth: Int, itemsPerPack: Int, itemWidth: Int, packsPerBlock: Int) =
    this(avalonDataWidth, wAddrWidth, itemsPerPack, itemWidth, packsPerBlock, itemWidth)

  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8 && avalonDataWidth >= itemWidthRaw)
  require(avalonDataWidth % itemWidthRaw == 0)
  require(itemWidthRaw >= itemWidth)

  val readsPerPack = itemsPerPack * itemWidthRaw / avalonDataWidth
  val packDelay = if (readsPerPack == 1) 0 else readsPerPack

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

  def unpackRead(packed: UInt) = {
    packed
  }

  object State {
    val idle :: running :: Nil = Enum(2)
  }

  val state = RegInit(State.idle)

  val idle = (state === State.idle)
  val running = (state === State.running)

  val readsCountLeft = if (packDelay == 0) 0.U else Reg(UInt(Chisel.log2Up(readsPerPack).W))
  val readsCountLast = (readsCountLeft === 0.U)
  if (packDelay != 0) {
    when(idle) {
      readsCountLeft := (readsPerPack - 1).U
    }.elsewhen(running & io.avalonMasterReadDataValid) {
      readsCountLeft := readsCountLeft - 1.U
    }
  }

  val packDone = if (packDelay == 0) io.avalonMasterReadDataValid else RegNext(readsCountLast & io.avalonMasterReadDataValid)
  val acceptData = (running & packDone)

  val packsCountLeft = Reg(UInt(Chisel.log2Up(packsPerBlock).W))
  val packsCountLast = (packsCountLeft === 0.U) & packDone
  when(idle) {
    packsCountLeft := (packsPerBlock - 1).U
  }.elsewhen(acceptData) {
    packsCountLeft := packsCountLeft - 1.U
  }

  val memAddress = RegInit(0.U(wAddrWidth.W))
  when(acceptData) {
    memAddress := memAddress + 1.U
  }

  val acceptLast = (acceptData & packsCountLast)
  when(idle & io.requesterNext) {
    state := State.running
  }.elsewhen(acceptLast) {
    state := State.idle
  }

  val buffer = ShiftingBuffer(packDelay, unpackRead(io.avalonMasterReadData), io.avalonMasterReadDataValid)

  io.memWrite.addr := memAddress
  io.memWrite.data := Seq.tabulate(itemsPerPack){i =>
    val itemMsb = (i + 1) * itemWidth - 1
    val itemLsb = i * itemWidth
    buffer(itemMsb, itemLsb)
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
