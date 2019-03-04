package bxb.adma

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyTile(val startAddress: Int, val height: Int, val width: Int, val rowToRowDistance: Int) {
}

class TileGeneratorParameters(b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) {
  private def divRoundUp(x: Int, y: Int) = {
    (x + y - 1) / y
  }

  val pad = 1
  val dep = 2

  require(tileHeight > dep && tileWidth > dep)

  val bytesPerElement = avalonDataWidth / 8

  val hCount = divRoundUp(inputHeight + 2 * pad - dep, tileHeight - dep)
  val wCount = divRoundUp(inputWidth + 2 * pad - dep, tileWidth - dep)
  val cCount = inputChannels / b

  require(hCount >= 3 && wCount >= 3, "tile generator assumes that at least 3 tiles exist in each direction")

  val leftTileW = tileWidth - pad
  val middleTileW = tileWidth
  val rightTileW = inputWidth + pad - (wCount - 1) * (tileWidth - dep)

  val topTileH = tileHeight - pad
  val middleTileH = tileHeight
  val bottomTileH = inputHeight + pad - (hCount - 1)  * (tileHeight - dep)

  val leftStep = leftTileW - dep
  val middleStep = middleTileW - dep

  val topRowDistance = inputWidth * (topTileH - dep) - inputWidth + rightTileW
  val midRowDistance = inputWidth * (middleTileH - dep) - inputWidth + rightTileW

  val leftRowToRowDistance = inputWidth - leftTileW + (if (leftTileW % maxBurst == 0) maxBurst else leftTileW % maxBurst)
  val middleRowToRowDistance = inputWidth - middleTileW + (if (middleTileW % maxBurst == 0) maxBurst else middleTileW % maxBurst)
  val rightRowToRowDistance = inputWidth - rightTileW + (if (rightTileW % maxBurst == 0) maxBurst else rightTileW % maxBurst)

  val inputSpace = inputHeight * inputWidth

  val topBottomLeftPad = (leftTileW + pad) * pad
  val topBottomMiddlePad = middleTileW * pad
  val topBottomRightPad = (rightTileW + pad) * pad
  val sidePad = pad

  def referenceTileSequence() = {
    val tileSeq = mutable.ArrayBuffer[DummyTile]()
    var startAddr = 0
    for (h <- 0 until hCount) {
      for (w <- 0 until wCount) {
        val verticalTop = (h == 0)
        val verticalMiddle = (h != hCount - 1)
        val horizontalLeft = (w == 0)
        val horizontalMiddle = (w != wCount - 1)
        val height = if (verticalTop) topTileH else if (verticalMiddle) middleTileH else bottomTileH
        val width = if (horizontalLeft) leftTileW else if (horizontalMiddle) middleTileW else rightTileW
        val rowToRowDist = if (horizontalLeft) leftRowToRowDistance else if (horizontalMiddle) middleRowToRowDistance else rightRowToRowDistance
        for (c <- 0 until cCount) {
          tileSeq += new DummyTile(startAddr + c * inputSpace * bytesPerElement, height, width, rowToRowDist)
        }
        if (horizontalLeft) {
          startAddr += leftStep * bytesPerElement
        }
        else if (horizontalMiddle) {
          startAddr += middleStep * bytesPerElement
        }
        else {
          startAddr += (if (verticalTop) topRowDistance else midRowDistance) * bytesPerElement
        }
      }
    }
    tileSeq
  }
}

class DummyTileGenerator(b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) {
  val tileSeq = mutable.ArrayBuffer[DummyTile]()

  val pad = 1
  val dep = 2

  val bytesPerElement = avalonDataWidth / 8

  for (tileY <- -pad until (inputHeight) by (tileHeight - dep)) {
    for (tileX <- -pad until (inputWidth) by (tileWidth - dep)) {
      for (tileC <- 0 until inputChannels by b) {
        val dataY = if (tileY < 0) 0 else tileY
        val dataEndY = if (tileY + tileHeight > inputHeight) inputHeight else tileY + tileHeight
        val dataHeight = dataEndY - dataY

        val dataX = if (tileX < 0) 0 else tileX
        val dataEndX = if (tileX + tileWidth > inputWidth) inputWidth else tileX + tileWidth
        val dataWidth = dataEndX - dataX

        if (dataWidth + pad > dep && dataHeight + pad > dep) {
          val dataAddr = (tileC / b * inputHeight * inputWidth + dataY * inputWidth + dataX) * bytesPerElement
          val rowToRowDist = inputWidth - dataWidth + (if (dataWidth % maxBurst == 0) maxBurst else dataWidth % maxBurst)
          tileSeq += new DummyTile(dataAddr, dataHeight, dataWidth, rowToRowDist)
        }
      }
    }
  }
}

class ADmaTileGeneratorTestSequence(dut: ADmaTileGenerator, b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyTileGenerator(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  poke(dut.io.start, true)

  poke(dut.io.inputAddress, 0)
  poke(dut.io.inputHCount, param.hCount)
  poke(dut.io.inputWCount, param.wCount)
  poke(dut.io.inputCCount, param.cCount)

  poke(dut.io.topTileH, param.topTileH)
  poke(dut.io.middleTileH, param.middleTileH)
  poke(dut.io.bottomTileH, param.bottomTileH)

  poke(dut.io.leftTileW, param.leftTileW)
  poke(dut.io.middleTileW, param.middleTileW)
  poke(dut.io.rightTileW, param.rightTileW)

  poke(dut.io.leftRowToRowDistance, param.leftRowToRowDistance)
  poke(dut.io.middleRowToRowDistance, param.middleRowToRowDistance)
  poke(dut.io.rightRowToRowDistance, param.rightRowToRowDistance)

  poke(dut.io.leftStep, param.leftStep)
  poke(dut.io.middleStep, param.middleStep)
  poke(dut.io.topRowDistance, param.topRowDistance)
  poke(dut.io.midRowDistance, param.midRowDistance)

  poke(dut.io.inputSpace, param.inputSpace)
  poke(dut.io.aWarZero, false)

  poke(dut.io.tileAccepted, false)
  for (tile <- ref.tileSeq) {
    println(f"startAddress: ${tile.startAddress}, height: ${tile.height}, width: ${tile.width}, rowToRowDistance: ${tile.rowToRowDistance}")
    while (peek(dut.io.tileValid) == 0) {
      step(1)
    }
    expect(dut.io.tileStartAddress, tile.startAddress)
    expect(dut.io.tileHeight, tile.height)
    expect(dut.io.tileWidth, tile.width)
    expect(dut.io.tileRowToRowDistance, tile.rowToRowDistance)
    poke(dut.io.tileAccepted, true)
    step(1)
    poke(dut.io.tileAccepted, false)
  }
}

class ADmaTileGeneratorTestAWarZero(dut: ADmaTileGenerator, b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyTileGenerator(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  poke(dut.io.start, true)

  poke(dut.io.inputAddress, 0)
  poke(dut.io.inputHCount, param.hCount)
  poke(dut.io.inputWCount, param.wCount)
  poke(dut.io.inputCCount, param.cCount)

  poke(dut.io.topTileH, param.topTileH)
  poke(dut.io.middleTileH, param.middleTileH)
  poke(dut.io.bottomTileH, param.bottomTileH)

  poke(dut.io.leftTileW, param.leftTileW)
  poke(dut.io.middleTileW, param.middleTileW)
  poke(dut.io.rightTileW, param.rightTileW)

  poke(dut.io.leftRowToRowDistance, param.leftRowToRowDistance)
  poke(dut.io.middleRowToRowDistance, param.middleRowToRowDistance)
  poke(dut.io.rightRowToRowDistance, param.rightRowToRowDistance)

  poke(dut.io.leftStep, param.leftStep)
  poke(dut.io.middleStep, param.middleStep)
  poke(dut.io.topRowDistance, param.topRowDistance)
  poke(dut.io.midRowDistance, param.midRowDistance)

  poke(dut.io.inputSpace, param.inputSpace)
  poke(dut.io.aWarZero, false)

  var acceptDelay = 1
  poke(dut.io.tileAccepted, false)
  for (tile <- ref.tileSeq) {
    println(f"startAddress: ${tile.startAddress}, height: ${tile.height}, width: ${tile.width}, rowToRowDistance: ${tile.rowToRowDistance}")
    poke(dut.io.aWarZero, true)
    for (_ <- 0 until acceptDelay) {
      step(1)
    }
    acceptDelay = (acceptDelay + 2) % 5
    expect(dut.io.tileValid, false)
    poke(dut.io.aWarZero, false)
    while (peek(dut.io.tileValid) == 0) {
      step(1)
    }
    expect(dut.io.tileStartAddress, tile.startAddress)
    expect(dut.io.tileHeight, tile.height)
    expect(dut.io.tileWidth, tile.width)
    expect(dut.io.tileRowToRowDistance, tile.rowToRowDistance)
    poke(dut.io.tileAccepted, true)
    step(1)
    poke(dut.io.tileAccepted, false)
  }
}

object ADmaTileGeneratorTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val inputWidth = 20
    val inputHeight = 20
    val inputChannels = 1 * b
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val tileCountWidth = aAddrWidth
    val avalonAddrWidth = Chisel.log2Up(inputWidth * inputWidth * inputChannels * b * 2 / 8)

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    val tileHeight = 5
    val tileWidth = 5

    val maxBurst = 4
    val avalonDataWidth = b * 2

    for (maxBurst <- List(1, 2, 4)) {
      for ((tileHeight, tileWidth) <- List((4, 4), (5, 5), (10, 10))) {
        ok &= Driver.execute(driverArgs, () => new ADmaTileGenerator(avalonAddrWidth, b * 2, tileCountWidth))(
          dut => new ADmaTileGeneratorTestSequence(dut, b, b * 2, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, 4))
        ok &= Driver.execute(driverArgs, () => new ADmaTileGenerator(avalonAddrWidth, b * 2, tileCountWidth))(
          dut => new ADmaTileGeneratorTestAWarZero(dut, b, b * 2, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, 4))
      }
    }
  }
}
