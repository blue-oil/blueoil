package bxb.a2f

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class A2fTileGenerator(tileCountWidth: Int) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())

    // Tile generation parameters
    // - should be equal to roundUp(outpuHeight / tileHeight)
    val outputHCount = Input(UInt(6.W))
    // - should be equal to roundUp(outputWidth / tileWidth)
    val outputWCount = Input(UInt(6.W))
    // - should be equeal to roundUp(outputChannels / B)
    val outputCCount = Input(UInt(6.W))

    // tileHeight
    val regularTileH = Input(UInt(tileCountWidth.W))
    // - outputHeight - (hCount - 1)  * tileHeight
    val lastTileH = Input(UInt(tileCountWidth.W))

    // tileWidth
    val regularTileW = Input(UInt(tileCountWidth.W))
    // - outputWidth - (wCount - 1)  * tileWidth
    val lastTileW = Input(UInt(tileCountWidth.W))

    // Tile output interface with handshaking
    val tileHeight = Output(UInt(tileCountWidth.W))
    val tileWidth = Output(UInt(tileCountWidth.W))
    val tileFirst = Output(Bool())
    val tileValid = Output(Bool())
    val tileAccepted = Input(Bool())

    // Status
    val statusReady = Output(Bool())
  })

  // FIXME: it is too much for the task done
  object State {
    val idle :: resetCounters :: updateCounters :: setupTile :: valid :: Nil = Enum(5)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val resetCounters = (state === State.resetCounters)
  val updateCounters = (state === State.updateCounters)
  val setupTile = (state === State.setupTile)
  val valid = (state === State.valid)

  val tileFirst = Reg(Bool())
  when(idle) {
    tileFirst := true.B
  }.elsewhen(valid & io.tileAccepted) {
    tileFirst := false.B
  }

  val outputCCountLeft = Reg(UInt(tileCountWidth.W))
  val outputCCountLast = (outputCCountLeft === 1.U)
  when(resetCounters) {
    outputCCountLeft := io.outputCCount
  }.elsewhen(updateCounters) {
    when(outputCCountLast) {
      outputCCountLeft := io.outputCCount
    }.otherwise {
      outputCCountLeft := outputCCountLeft - 1.U
    }
  }

  val outputWCountLeft = Reg(UInt(tileCountWidth.W))
  val outputWCountOne = (outputWCountLeft === 1.U)
  val outputWCountLast = outputWCountOne & outputCCountLast
  when(resetCounters) {
    outputWCountLeft := io.outputWCount
  }.elsewhen(updateCounters & outputCCountLast) {
    when(outputWCountLast) {
      outputWCountLeft := io.outputWCount
    }.otherwise {
      outputWCountLeft := outputWCountLeft - 1.U
    }
  }
  val horizontalLast = outputWCountOne

  val outputHCountLeft = Reg(UInt(tileCountWidth.W))
  val outputHCountOne = (outputHCountLeft === 1.U)
  val outputHCountLast = outputHCountOne & outputWCountLast
  when(resetCounters) {
    outputHCountLeft := io.outputHCount
  }.elsewhen(updateCounters & outputWCountLast) {
    when(outputHCountLast) {
      outputHCountLeft := io.outputHCount
    }.otherwise {
      outputHCountLeft := outputHCountLeft - 1.U
    }
  }
  val verticalLast = outputHCountOne

  val tileWidth = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(~horizontalLast) {
      tileWidth := io.regularTileW
    }.otherwise {
      tileWidth := io.lastTileW
    }
  }

  val tileHeight = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(~verticalLast) {
      tileHeight := io.regularTileH
    }.otherwise {
      tileHeight := io.lastTileH
    }
  }

  when(idle & io.start) {
    state := State.resetCounters
  }.elsewhen(resetCounters) {
    state := State.setupTile
  }.elsewhen(updateCounters) {
    state := State.setupTile
  }.elsewhen(setupTile) {
    state := State.valid
  }.elsewhen(valid & io.tileAccepted) {
    when(outputHCountLast) {
      state := State.idle
    }.otherwise {
      state := State.updateCounters
    }
  }

  io.tileHeight := tileHeight
  io.tileWidth := tileWidth
  io.tileFirst := tileFirst
  io.tileValid := valid
  io.statusReady := idle
}

object A2fTileGenerator {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new A2fTileGenerator(12)))
  }
}
