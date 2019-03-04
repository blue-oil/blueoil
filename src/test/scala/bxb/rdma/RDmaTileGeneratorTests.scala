package bxb.rdma

import scala.collection._
import scala.math.{min}
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyTile(val startAddress: Int, val height: Int, val width: Int, val rowToRowDistance: Int) {
}

class TileGeneratorParameters(b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) {
  private def divRoundUp(x: Int, y: Int) = {
    (x + y - 1) / y
  }

  val bytesPerElement = avalonDataWidth / 8

  val hCount = divRoundUp(outputHeight, tileHeight)
  val wCount = divRoundUp(outputWidth, tileWidth)
  val cCount = outputChannels / b

  val regularTileH = tileHeight
  val lastTileH = outputHeight - (hCount - 1)  * tileHeight

  val regularTileW = tileWidth
  val lastTileW = outputWidth - (wCount - 1)  * tileWidth

  val regularRowToRowDistance = outputWidth - regularTileW + (if (regularTileW % maxBurst == 0) maxBurst else regularTileW % maxBurst)
  val lastRowToRowDistance = outputWidth - lastTileW + (if (lastTileW % maxBurst == 0) maxBurst else lastTileW % maxBurst)

  val outputSpace = outputHeight * outputWidth

  val rowDistance = outputWidth * regularTileH - outputWidth + lastTileW

  def referenceTileSequence() = {
    val tileSeq = mutable.ArrayBuffer[DummyTile]()
    var startAddr = 0
    for (h <- 0 until hCount) {
      for (w <- 0 until wCount) {
        val verticalLast = (h == hCount - 1)
        val horizontalLast = (w == wCount - 1)
        val height = if (!verticalLast) regularTileH else lastTileH
        val width = if (!horizontalLast) regularTileW else lastTileW
        val rowToRowDist = if (!horizontalLast) regularRowToRowDistance else lastRowToRowDistance
        for (c <- 0 until cCount) {
          tileSeq += new DummyTile(startAddr + c * outputSpace * bytesPerElement, height, width, rowToRowDist)
        }
        if (!horizontalLast) {
          startAddr += regularTileW * bytesPerElement
        }
        else {
          startAddr += rowDistance * bytesPerElement 
        }
      }
    }
    tileSeq
  }
}

class DummyTileGenerator(b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) {
  val tileSeq = mutable.ArrayBuffer[DummyTile]()

  val bytesPerElement = avalonDataWidth / 8

  for (tileY <- 0 until outputHeight by tileHeight) {
    for (tileX <- 0 until outputWidth by tileWidth) {
      for (tileC <- 0 until outputChannels by b) {
        val tileEndY = min(tileY + tileHeight, outputHeight)
        val height = tileEndY - tileY

        val tileEndX = min(tileX + tileWidth, outputWidth)
        val width = tileEndX - tileX

        val tileAddr = (tileC / b * outputHeight * outputWidth + tileY * outputWidth + tileX) * bytesPerElement
        val rowToRowDist = outputWidth - width + (if (width % maxBurst == 0) maxBurst else width % maxBurst)
        tileSeq += new DummyTile(tileAddr, height, width, rowToRowDist)
      }
    }
  }
}

class RDmaTileGeneratorTestSequence(dut: RDmaTileGenerator, b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyTileGenerator(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  poke(dut.io.start, true)

  poke(dut.io.outputAddress, 0)
  poke(dut.io.outputHCount, param.hCount)
  poke(dut.io.outputWCount, param.wCount)
  poke(dut.io.outputCCount, param.cCount)

  poke(dut.io.regularTileH, param.regularTileH)
  poke(dut.io.lastTileH, param.lastTileH)

  poke(dut.io.regularTileW, param.regularTileW)
  poke(dut.io.lastTileW, param.lastTileW)

  poke(dut.io.regularRowToRowDistance, param.regularRowToRowDistance)
  poke(dut.io.lastRowToRowDistance, param.lastRowToRowDistance)

  poke(dut.io.rowDistance, param.rowDistance)

  poke(dut.io.outputSpace, param.outputSpace)
  poke(dut.io.rRawZero, false)

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

class RDmaTileGeneratorTestRRawZero(dut: RDmaTileGenerator, b: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyTileGenerator(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  poke(dut.io.start, true)

  poke(dut.io.outputAddress, 0)
  poke(dut.io.outputHCount, param.hCount)
  poke(dut.io.outputWCount, param.wCount)
  poke(dut.io.outputCCount, param.cCount)

  poke(dut.io.regularTileH, param.regularTileH)
  poke(dut.io.lastTileH, param.lastTileH)

  poke(dut.io.regularTileW, param.regularTileW)
  poke(dut.io.lastTileW, param.lastTileW)

  poke(dut.io.regularRowToRowDistance, param.regularRowToRowDistance)
  poke(dut.io.lastRowToRowDistance, param.lastRowToRowDistance)

  poke(dut.io.rowDistance, param.rowDistance)

  poke(dut.io.outputSpace, param.outputSpace)
  poke(dut.io.rRawZero, false)

  var acceptDelay = 1
  poke(dut.io.tileAccepted, false)
  for (tile <- ref.tileSeq) {
    println(f"startAddress: ${tile.startAddress}, height: ${tile.height}, width: ${tile.width}, rowToRowDistance: ${tile.rowToRowDistance}")
    poke(dut.io.rRawZero, true)
    for (_ <- 0 until acceptDelay) {
      step(1)
    }
    acceptDelay = (acceptDelay + 2) % 5
    expect(dut.io.tileValid, false)
    poke(dut.io.rRawZero, false)
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

object RDmaTileGeneratorTests {
  def main(args: Array[String]): Unit = {
    val b = 4
    val outputWidth = 10
    val outputHeight = 10
    val outputChannels = 2 * b
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val tileCountWidth = aAddrWidth
    val avalonAddrWidth = Chisel.log2Up(outputWidth * outputWidth * outputChannels * b * 2 / 8)

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    val avalonDataWidth = b * 2

    breakable {
      for (maxBurst <- List(1, 2, 4)) {
        for ((tileHeight, tileWidth) <- List((4, 4), (5, 5), (10, 10))) {
          ok &= Driver.execute(driverArgs, () => new RDmaTileGenerator(avalonAddrWidth, b * 2, tileCountWidth))(
            dut => new RDmaTileGeneratorTestSequence(dut, b, b * 2, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, 4))
          ok &= Driver.execute(driverArgs, () => new RDmaTileGenerator(avalonAddrWidth, b * 2, tileCountWidth))(
            dut => new RDmaTileGeneratorTestRRawZero(dut, b, b * 2, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, 4))
          if (!ok) {
            println(f"Failed for maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
            break
          }
        }
      }
    }
  }
}
