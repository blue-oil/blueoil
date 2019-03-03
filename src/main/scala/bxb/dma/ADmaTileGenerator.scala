package bxb.dma

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class ADmaTileGenerator(avalonAddrWidth: Int, avalonDataWidth: Int, tileCountWidth: Int) extends Module {
  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  val avalonDataByteWidth = avalonDataWidth / 8
  val avalonDataByteWidthLog = Chisel.log2Ceil(avalonDataByteWidth)
  val io = IO(new Bundle {
    val start = Input(Bool())

    // Tile generation parameters
    val inputAddress = Input(UInt(avalonAddrWidth.W))
    // - should be equal to roundUp(inputHeight / (tileHeight - pad))
    val inputHCount = Input(UInt(6.W))
    // - should be equal to roundUp(inputWidth / (tileWidth - pad))
    val inputWCount = Input(UInt(6.W))
    // - should be equal to roundUp(inputChannels / B)
    val inputCCount = Input(UInt(6.W))

    // - tileHeight - pad
    val topTileH = Input(UInt(tileCountWidth.W))
    // - tileHeight
    val middleTileH = Input(UInt(tileCountWidth.W))
    // - inputHeight + pad - (hCount - 1)  * (tileHeight - pad)
    val bottomTileH = Input(UInt(tileCountWidth.W))

    // - tileWidth - pad
    val leftTileW = Input(UInt(tileCountWidth.W))
    // - tileWidth
    val middleTileW = Input(UInt(tileCountWidth.W))
    // - inputWidth + pad - (wCount - 1) * (tileWidth - pad)
    val rightTileW = Input(UInt(tileCountWidth.W))

    // (inputWidth - leftTileW + (leftTileW % maxBurst == 0) ? maxBurst : leftTileW % maxBurst)
    val leftRowToRowDistance = Input(UInt(tileCountWidth.W))
    // (inputWidth - middleTileW + (middleTileW % maxBurst == 0) ? maxBurst : middleTileW % maxBurst)
    val middleRowToRowDistance = Input(UInt(tileCountWidth.W))
    // (inputWidth - rightTileW + (rightTileW % maxBurst == 0) ? maxBurst : rightTileW % maxBurst)
    val rightRowToRowDistance = Input(UInt(tileCountWidth.W))

    // leftTileW - pad
    val leftStep = Input(UInt(avalonAddrWidth.W))
    // middleTileW - pad
    val middleStep = Input(UInt(avalonAddrWidth.W))

    // inputWidth * (topTileH - pad) - inputWidth + rightTileW
    val topRowDistance = Input(UInt(avalonAddrWidth.W))
    // inputWidth * (middleTileH - pad) - inputWidth + rightTileW
    val midRowDistance = Input(UInt(avalonAddrWidth.W))

    // inputWidth * inputHeight
    val inputSpace = Input(UInt(avalonAddrWidth.W))

    // (leftTileW + pad) * pad
    val topBottomLeftPad = Input(UInt(tileCountWidth.W))
    // middleTileW * pad
    val topBottomMiddlePad = Input(UInt(tileCountWidth.W))
    // (rightTileW + pad) * pad
    val topBottomRightPad = Input(UInt(tileCountWidth.W))
    // pad
    val sidePad = Input(UInt(tileCountWidth.W))

    // Tile output interface with handshaking
    val tileStartAddress = Output(UInt(avalonAddrWidth.W))
    val tileHeight = Output(UInt(tileCountWidth.W))
    val tileWidth = Output(UInt(tileCountWidth.W))
    val tileRowToRowDistance = Output(UInt(tileCountWidth.W))
    val tileValid = Output(Bool())
    val tileAccepted = Input(Bool())
    val tileStartPad = Output(UInt(tileCountWidth.W))
    val tileSidePad = Output(UInt(tileCountWidth.W))
    val tileEndPad = Output(UInt(tileCountWidth.W))

    // Synchronization interface
    val aWarDec = Output(Bool())
    val aWarZero = Input(Bool())
  })

  private def toBytes(elements: UInt) = {
    elements << avalonDataByteWidthLog.U
  }

  object State {
    val idle :: resetCounters :: updateCounters :: setupTile :: waitSync :: valid :: Nil = Enum(6)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val resetCounters = (state === State.resetCounters)
  val updateCounters = (state === State.updateCounters)
  val setupTile = (state === State.setupTile)
  val waitSync = (state === State.waitSync)
  val valid = (state === State.valid)

  val syncDecAWar = (setupTile | waitSync)

  val inputCCountLeft = Reg(UInt(tileCountWidth.W))
  val inputCCountLast = (inputCCountLeft === 1.U)
  when(resetCounters) {
    inputCCountLeft := io.inputCCount
  }.elsewhen(updateCounters) {
    when(inputCCountLast) {
      inputCCountLeft := io.inputCCount
    }.otherwise {
      inputCCountLeft := inputCCountLeft - 1.U
    }
  }

  object DepthState {
    val first :: rest :: Nil = Enum(2)
  }

  val depthState = RegInit(DepthState.first)
  val depthFirst = (depthState === DepthState.first)
  when(resetCounters) {
    depthState := DepthState.first
  }.elsewhen(updateCounters) {
    when(inputCCountLast) {
      depthState := DepthState.first
    }.elsewhen(depthFirst) {
      depthState := DepthState.rest
    }
  }

  val inputWCountLeft = Reg(UInt(tileCountWidth.W))
  val inputWCountOne = (inputWCountLeft === 1.U)
  val inputWCountLast = inputWCountOne & inputCCountLast
  when(resetCounters) {
    inputWCountLeft := io.inputWCount
  }.elsewhen(updateCounters & inputCCountLast) {
    when(inputWCountLast) {
      inputWCountLeft := io.inputWCount
    }.otherwise {
      inputWCountLeft := inputWCountLeft - 1.U
    }
  }

  object HorizontalState {
    val left :: middle :: Nil = Enum(2)
  }

  val horizontalState = RegInit(HorizontalState.left)
  val horizontalLeft = (horizontalState === HorizontalState.left)
  val horizontalRight = inputWCountOne
  val horizontalMiddle = (horizontalState === HorizontalState.middle) & ~horizontalRight

  when(resetCounters) {
    horizontalState := HorizontalState.left
  }.elsewhen(updateCounters & inputCCountLast) {
    when(inputWCountLast) {
      horizontalState := HorizontalState.left
    }.elsewhen(horizontalLeft) {
      horizontalState := HorizontalState.middle
    }
  }

  val inputHCountLeft = Reg(UInt(tileCountWidth.W))
  val inputHCountOne = (inputHCountLeft === 1.U)
  val inputHCountLast = inputHCountOne & inputWCountLast
  when(resetCounters) {
    inputHCountLeft := io.inputHCount
  }.elsewhen(updateCounters & inputWCountLast) {
    when(inputHCountLast) {
      inputHCountLeft := io.inputHCount
    }.otherwise {
      inputHCountLeft := inputHCountLeft - 1.U
    }
  }

  object VerticalState {
    val top :: middle :: Nil = Enum(2)
  }

  val verticalState = RegInit(VerticalState.top)
  val verticalTop = (verticalState === VerticalState.top)
  val verticalBottom = inputHCountOne
  val verticalMiddle = (verticalState === VerticalState.middle) & ~verticalBottom

  when(resetCounters) {
    verticalState := VerticalState.top
  }.elsewhen(updateCounters & inputWCountLast) {
    when(inputHCountOne) {
      verticalState := VerticalState.top
    }.elsewhen(verticalTop) {
      verticalState := VerticalState.middle
    }
  }

  val tileFirstChannelAddress = Reg(UInt(avalonAddrWidth.W))
  when(resetCounters) {
    tileFirstChannelAddress := io.inputAddress
  }.elsewhen(updateCounters & inputCCountLast) {
    when(horizontalLeft) {
      tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.leftStep)
    }.elsewhen(horizontalMiddle) {
      tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.middleStep)
    }.otherwise {
      when(verticalTop) {
        tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.topRowDistance)
      }.otherwise {
        tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.midRowDistance)
      }
    }
  }

  val tileAddress = Reg(UInt(avalonAddrWidth.W))
  when(setupTile) {
    when(depthFirst) {
      // printf(p"tileAddress := ${tileFirstChannelAddress}\n")
      tileAddress := tileFirstChannelAddress
    }.otherwise {
      // printf(p"tileAddress := ${tileAddress} + ${toBytes(io.inputSpace)}\n")
      tileAddress := tileAddress + toBytes(io.inputSpace)
    }
  }

  val tileWidth = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(horizontalLeft) {
      tileWidth := io.leftTileW
    }.elsewhen(horizontalMiddle) {
      tileWidth := io.middleTileW
    }.otherwise {
      tileWidth := io.rightTileW
    }
  }

  val tileHeight = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(verticalTop) {
      tileHeight := io.topTileH
    }.elsewhen(verticalMiddle) {
      tileHeight := io.middleTileH
    }.otherwise {
      tileHeight := io.bottomTileH
    }
  }

  val tileRowToRowDistance = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(horizontalLeft) {
      tileRowToRowDistance := io.leftRowToRowDistance
    }.elsewhen(horizontalMiddle) {
      tileRowToRowDistance := io.middleRowToRowDistance
    }.otherwise {
      tileRowToRowDistance := io.rightRowToRowDistance
    }
  }

  val tileStartPad = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(verticalTop) {
      when(horizontalLeft) {
        tileStartPad := io.topBottomLeftPad + io.sidePad
      }.elsewhen(horizontalMiddle) {
        tileStartPad := io.topBottomMiddlePad
      }.otherwise {
        tileStartPad := io.topBottomRightPad
      }
    }.elsewhen(horizontalLeft) {
      tileStartPad := io.sidePad
    }.otherwise {
      tileStartPad := 0.U
    }
  }

  val tileSidePad = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    tileSidePad := Mux(horizontalMiddle, 0.U, io.sidePad)
  }

  val tileEndPad = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(verticalBottom) {
      when(horizontalLeft) {
        tileEndPad := io.topBottomLeftPad
      }.elsewhen(horizontalMiddle) {
        tileEndPad := io.topBottomMiddlePad
      }.otherwise {
        tileEndPad := io.topBottomRightPad + io.sidePad
      }
    }.elsewhen(horizontalRight) {
      tileEndPad := io.sidePad
    }.otherwise {
      tileEndPad := 0.U
    }
  }

  when(idle & io.start) {
    state := State.resetCounters
  }.elsewhen(resetCounters) {
    state := State.setupTile
  }.elsewhen(updateCounters) {
    state := State.setupTile
  }.elsewhen(setupTile & io.aWarZero) {
    state := State.waitSync
  }.elsewhen((setupTile | waitSync) & ~io.aWarZero) {
    state := State.valid
  }.elsewhen(valid & io.tileAccepted) {
    state := State.updateCounters
  }

  io.tileStartAddress := tileAddress
  io.tileHeight := tileHeight
  io.tileWidth := tileWidth
  io.tileRowToRowDistance := tileRowToRowDistance
  io.tileValid := valid
  io.tileStartPad := tileStartPad
  io.tileSidePad := tileSidePad
  io.tileEndPad := tileEndPad
  io.aWarDec := syncDecAWar
}

object ADmaTileGenerator {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new ADmaTileGenerator(32, 64, 12)))
  }
}
