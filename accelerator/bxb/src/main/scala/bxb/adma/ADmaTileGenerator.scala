package bxb.adma

import chisel3._
import chisel3.util._

import bxb.util.{Util}

class ADmaParameters(private val avalonAddrWidth: Int, private val tileCountWidth: Int) extends Bundle {
  // Tile generation parameters
  val inputAddress = UInt(avalonAddrWidth.W)
  // - should be equal to roundUp(inputHeight / (tileHeight - pad))
  val inputHCount = UInt(6.W)
  // - should be equal to roundUp(inputWidth / (tileWidth - pad))
  val inputWCount = UInt(6.W)
  // - should be equal to roundUp(inputChannels / B)
  val inputCCount = UInt(6.W)
  // - should be equal to roundUp(outputChannels / B)
  val outputCCount = UInt(6.W)

  // - tileHeight - pad
  val topTileH = UInt(tileCountWidth.W)
  // - tileHeight
  val middleTileH = UInt(tileCountWidth.W)
  // - inputHeight + pad - (hCount - 1)  * (tileHeight - pad)
  val bottomTileH = UInt(tileCountWidth.W)

  // - tileWidth - pad
  val leftTileW = UInt(tileCountWidth.W)
  // - tileWidth
  val middleTileW = UInt(tileCountWidth.W)
  // - inputWidth + pad - (wCount - 1) * (tileWidth - pad)
  val rightTileW = UInt(tileCountWidth.W)

  // (inputWidth - leftTileW + (leftTileW % maxBurst == 0) ? maxBurst : leftTileW % maxBurst)
  val leftRowToRowDistance = UInt(tileCountWidth.W)
  // (inputWidth - middleTileW + (middleTileW % maxBurst == 0) ? maxBurst : middleTileW % maxBurst)
  val middleRowToRowDistance = UInt(tileCountWidth.W)
  // (inputWidth - rightTileW + (rightTileW % maxBurst == 0) ? maxBurst : rightTileW % maxBurst)
  val rightRowToRowDistance = UInt(tileCountWidth.W)

  // leftTileW - pad
  val leftStep = UInt(avalonAddrWidth.W)
  // middleTileW - pad
  val middleStep = UInt(avalonAddrWidth.W)

  // inputWidth * (topTileH - pad) - inputWidth + rightTileW
  val topRowDistance = UInt(avalonAddrWidth.W)
  // inputWidth * (middleTileH - pad) - inputWidth + rightTileW
  val midRowDistance = UInt(avalonAddrWidth.W)

  // inputWidth * inputHeight
  val inputSpace = UInt(avalonAddrWidth.W)

  // (leftTileW + pad) * pad
  val topBottomLeftPad = UInt(tileCountWidth.W)
  // middleTileW * pad
  val topBottomMiddlePad = UInt(tileCountWidth.W)
  // (rightTileW + pad) * pad
  val topBottomRightPad = UInt(tileCountWidth.W)
  // pad
  val sidePad = UInt(tileCountWidth.W)
}

object ADmaParameters {
  def apply(avalonAddrWidth: Int, tileCountWidth: Int) = {
    new ADmaParameters(avalonAddrWidth, tileCountWidth)
  }
}

class ADmaTileGenerator(avalonAddrWidth: Int, avalonDataWidth: Int, tileCountWidth: Int) extends Module {
  require(isPow2(avalonDataWidth) && avalonDataWidth >= 8)
  val avalonDataByteWidth = avalonDataWidth / 8
  val avalonDataByteWidthLog = Chisel.log2Ceil(avalonDataByteWidth)
  val io = IO(new Bundle {
    val start = Input(Bool())

    // Tile generation parameters
    val parameters = Input(ADmaParameters(avalonAddrWidth, tileCountWidth))

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
    val tileFirst = Output(Bool())

    // Synchronization interface
    val aWarDec = Output(Bool())
    val aWarZero = Input(Bool())

    // Status
    val statusReady = Output(Bool())
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

  val tileFirst = Reg(Bool())
  when(idle) {
    tileFirst := true.B
  }.elsewhen(valid & io.tileAccepted) {
    tileFirst := false.B
  }

  val inputCCountLeft = Reg(UInt(tileCountWidth.W))
  val inputCCountLast = (inputCCountLeft === 1.U)
  when(resetCounters) {
    inputCCountLeft := io.parameters.inputCCount
  }.elsewhen(updateCounters) {
    when(inputCCountLast) {
      inputCCountLeft := io.parameters.inputCCount
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

  val outputCCountLeft = Reg(UInt(tileCountWidth.W))
  val outputCCountLast = (outputCCountLeft === 1.U) & inputCCountLast
  when(resetCounters) {
    outputCCountLeft := io.parameters.outputCCount
  }.elsewhen(updateCounters & inputCCountLast) {
    when(outputCCountLast) {
      outputCCountLeft := io.parameters.outputCCount
    }.otherwise {
      outputCCountLeft := outputCCountLeft - 1.U
    }
  }

  val inputWCountLeft = Reg(UInt(tileCountWidth.W))
  val inputWCountOne = (inputWCountLeft === 1.U)
  val inputWCountLast = inputWCountOne & outputCCountLast
  when(resetCounters) {
    inputWCountLeft := io.parameters.inputWCount
  }.elsewhen(updateCounters & outputCCountLast) {
    when(inputWCountLast) {
      inputWCountLeft := io.parameters.inputWCount
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
  }.elsewhen(updateCounters & outputCCountLast) {
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
    inputHCountLeft := io.parameters.inputHCount
  }.elsewhen(updateCounters & inputWCountLast) {
    when(inputHCountLast) {
      inputHCountLeft := io.parameters.inputHCount
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
    tileFirstChannelAddress := io.parameters.inputAddress
  }.elsewhen(updateCounters & outputCCountLast) {
    when(horizontalLeft) {
      tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.parameters.leftStep)
    }.elsewhen(horizontalMiddle) {
      tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.parameters.middleStep)
    }.otherwise {
      when(verticalTop) {
        tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.parameters.topRowDistance)
      }.otherwise {
        tileFirstChannelAddress := tileFirstChannelAddress + toBytes(io.parameters.midRowDistance)
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
      tileAddress := tileAddress + toBytes(io.parameters.inputSpace)
    }
  }

  val tileWidth = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(horizontalLeft) {
      tileWidth := io.parameters.leftTileW
    }.elsewhen(horizontalMiddle) {
      tileWidth := io.parameters.middleTileW
    }.otherwise {
      tileWidth := io.parameters.rightTileW
    }
  }

  val tileHeight = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(verticalTop) {
      tileHeight := io.parameters.topTileH
    }.elsewhen(verticalMiddle) {
      tileHeight := io.parameters.middleTileH
    }.otherwise {
      tileHeight := io.parameters.bottomTileH
    }
  }

  val tileRowToRowDistance = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(horizontalLeft) {
      tileRowToRowDistance := io.parameters.leftRowToRowDistance
    }.elsewhen(horizontalMiddle) {
      tileRowToRowDistance := io.parameters.middleRowToRowDistance
    }.otherwise {
      tileRowToRowDistance := io.parameters.rightRowToRowDistance
    }
  }

  val tileStartPad = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(verticalTop) {
      when(horizontalLeft) {
        tileStartPad := io.parameters.topBottomLeftPad + io.parameters.sidePad
      }.elsewhen(horizontalMiddle) {
        tileStartPad := io.parameters.topBottomMiddlePad
      }.otherwise {
        tileStartPad := io.parameters.topBottomRightPad
      }
    }.elsewhen(horizontalLeft) {
      tileStartPad := io.parameters.sidePad
    }.otherwise {
      tileStartPad := 0.U
    }
  }

  val tileSidePad = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    tileSidePad := Mux(horizontalMiddle, 0.U, io.parameters.sidePad)
  }

  val tileEndPad = Reg(UInt(tileCountWidth.W))
  when(setupTile) {
    when(verticalBottom) {
      when(horizontalRight) {
        tileEndPad := io.parameters.topBottomRightPad + io.parameters.sidePad
      }.elsewhen(horizontalMiddle) {
        tileEndPad := io.parameters.topBottomMiddlePad
      }.otherwise {
        tileEndPad := io.parameters.topBottomLeftPad
      }
    }.elsewhen(horizontalRight) {
      tileEndPad := io.parameters.sidePad
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
    when(inputHCountLast) {
      state := State.idle
    }.otherwise {
      state := State.updateCounters
    }
  }

  io.tileStartAddress := tileAddress
  io.tileHeight := tileHeight
  io.tileWidth := tileWidth
  io.tileRowToRowDistance := tileRowToRowDistance
  io.tileValid := valid
  io.tileStartPad := tileStartPad
  io.tileSidePad := tileSidePad
  io.tileEndPad := tileEndPad
  io.tileFirst := tileFirst
  io.aWarDec := syncDecAWar
  io.statusReady := idle
}

object ADmaTileGenerator {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new ADmaTileGenerator(32, 64, 12)))
  }
}
