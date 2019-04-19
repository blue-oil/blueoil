package bxb.rdma

import chisel3._
import chisel3.util._

import bxb.memory.{ReadPort}
import bxb.util.{Util, CatLeastFirst}

class RDmaBufferedRMemReader(b: Int, avalonDataWidth: Int, rAddrWidth: Int, tileCountWidth: Int) extends Module {
  val rSz = 2
  // current data format assumes 32 activations packed into two 32 bit words:
  // first contains 32 lsb bits, second contains 32 msb bits
  val rBitPackSize = 32
  val rAddrLowWidth = rAddrWidth - 1

  require(avalonDataWidth <= 256, "exceeds maximum size of hps sdram slave port")
  require(b >= rBitPackSize && avalonDataWidth >= 2 * rBitPackSize,
    "to simplify logic we assume that we could receive amount of data just right to write b elements in one cycle")
  require(avalonDataWidth == b * rSz)
  require(avalonDataWidth % rBitPackSize == 0)

  val io = IO(new Bundle {
    // Tile Generator interface
    val tileHeight = Input(UInt(rAddrWidth.W))
    val tileWidth = Input(UInt(rAddrWidth.W))
    val tileFirst = Input(Bool())
    // once tileValid asserted above tile parameters must remain stable and until tileAccepted is asserted
    val tileValid = Input(Bool())

    val rmemRead = Output(Vec(b, ReadPort(rAddrWidth)))
    val rmemQ = Input(Vec(b, UInt(rSz.W)))

    val waitRequest = Input(Bool())
    // reader is ready since first word arrives and untill last word is assigned to data output
    val ready = Output(Bool())

    val data = Output(UInt(avalonDataWidth.W))
  })

  object State {
    val idle :: askFirst :: running :: waiting :: lastRead :: acknowledge :: Nil = Enum(6)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val askFirst = (state === State.askFirst)
  val running = (state === State.running)
  val waiting = (state === State.waiting)
  val lastRead = (state === State.lastRead)
  val acknowledge = (state === State.acknowledge)

  val incrementAddress = (askFirst | running | waiting)

  val updateAddress = (idle | (incrementAddress & ~io.waitRequest))

  // tile loops
  val tileXCountLeft = Reg(UInt(tileCountWidth.W))
  val tileXCountLast = (tileXCountLeft === 1.U)
  when(updateAddress) {
    when(idle | tileXCountLast) {
      tileXCountLeft := io.tileWidth
    }.otherwise {
      tileXCountLeft := tileXCountLeft - 1.U
    }
  }

  val tileYCountLeft = Reg(UInt(tileCountWidth.W))
  val tileYCountLast = (tileYCountLeft === 1.U) & tileXCountLast
  when(updateAddress) {
    when(idle | tileYCountLast) {
      tileYCountLeft := io.tileHeight
    }.elsewhen(tileXCountLast) {
      tileYCountLeft := tileYCountLeft - 1.U
    }
  }

  val rmemAddressMsb = RegInit(0.U(1.W))
  when(idle & io.tileValid & io.tileFirst) {
    rmemAddressMsb := 0.U
  }.elsewhen(idle & io.tileValid) {
    rmemAddressMsb := ~rmemAddressMsb
  }

  val rmemAddressLow = Reg(UInt(rAddrLowWidth.W))
  when(updateAddress) {
    when(idle) {
      rmemAddressLow := 0.U
    }.elsewhen(~tileYCountLast) {
      rmemAddressLow := rmemAddressLow + 1.U
    }
  }

  for (row <- 0 until b) {
    io.rmemRead(row).addr := Cat(rmemAddressMsb, rmemAddressLow)
    io.rmemRead(row).enable := true.B
  }

  // I wish we will switch to less hacky representation before we start support higher bit activations
  require(rSz == 2, "below assumes aSz to be 2 bits for now")
  require(avalonDataWidth / rSz == b && avalonDataWidth % rSz == 0)
  require(avalonDataWidth % (rSz * rBitPackSize) == 0)
  // extract all msb of each rmemQ element and group extracted bits into 32 bit chunks
  val packedMsbs = Seq.tabulate(b)(row => io.rmemQ(row)(1)).grouped(rBitPackSize).map(bits => CatLeastFirst(bits)).toSeq
  // extract all lsbs of each rmemQ element and group extracted bits into 32 bit chunks
  val packedLsbs = Seq.tabulate(b)(row => io.rmemQ(row)(0)).grouped(rBitPackSize).map(bits => CatLeastFirst(bits)).toSeq
  // concatenate all the chunks in order {msbs, lsbs, msbs, lsbs, ...}
  val packedData = CatLeastFirst(packedMsbs.zip(packedLsbs).map{case (msbPack, lsbPack) => Cat(msbPack, lsbPack)})

  val waitBuffer = Reg(UInt(avalonDataWidth.W))
  when(~waiting) {
    waitBuffer := packedData
  }

  when(idle & io.tileValid) {
    state := State.askFirst
  }.elsewhen(incrementAddress & tileYCountLast & ~io.waitRequest) {
    state := State.lastRead
  }.elsewhen(askFirst) {
    state := State.running
  }.elsewhen(running & io.waitRequest) {
    state := State.waiting
  }.elsewhen(waiting & ~io.waitRequest) {
    state := State.running
  }.elsewhen(lastRead & ~io.waitRequest) {
    state := State.acknowledge
  }.elsewhen(acknowledge) {
    state := State.idle
  }

  io.ready := ~idle
  io.data := Mux(waiting, waitBuffer, packedData)
}

object RDmaBufferedRMemReader {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new RDmaBufferedRMemReader(32, 64, 12, 12)))
  }
}
