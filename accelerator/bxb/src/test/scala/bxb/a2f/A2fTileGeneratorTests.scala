package bxb.a2f

import scala.collection._
import scala.math.{min}
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyTile(val height: Int, val width: Int) {
}

class TileGeneratorParameters(b: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int) {
  private def divRoundUp(x: Int, y: Int) = {
    (x + y - 1) / y
  }

  val hCount = divRoundUp(outputHeight, tileHeight)
  val wCount = divRoundUp(outputWidth, tileWidth)
  val cOutCount = outputChannels / b

  val regularTileH = tileHeight
  val lastTileH = outputHeight - (hCount - 1)  * tileHeight

  val regularTileW = tileWidth
  val lastTileW = outputWidth - (wCount - 1)  * tileWidth

  def referenceTileSequence() = {
    val tileSeq = mutable.ArrayBuffer[DummyTile]()
    var startAddr = 0
    for (h <- 0 until hCount) {
      for (w <- 0 until wCount) {
        for (c <- 0 until cOutCount) {
          val verticalLast = (h == hCount - 1)
          val horizontalLast = (w == wCount - 1)
          val height = if (!verticalLast) regularTileH else lastTileH
          val width = if (!horizontalLast) regularTileW else lastTileW
          tileSeq += new DummyTile(height, width)
        }
      }
    }
    tileSeq
  }
}

class DummyTileGenerator(b: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int) {
  val tileSeq = mutable.ArrayBuffer[DummyTile]()

  for (tileY <- 0 until outputHeight by tileHeight) {
    for (tileX <- 0 until outputWidth by tileWidth) {
      for (tileOutC <- 0 until outputChannels by b) {
        val tileEndY = min(tileY + tileHeight, outputHeight)
        val height = tileEndY - tileY

        val tileEndX = min(tileX + tileWidth, outputWidth)
        val width = tileEndX - tileX
        tileSeq += new DummyTile(height, width)
      }
    }
  }
}

class A2fTileGeneratorTestSequence(dut: A2fTileGenerator, b: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, outputChannels: Int) extends PeekPokeTester(dut) {
  val ref = new DummyTileGenerator(b, tileHeight, tileWidth, inputHeight, inputWidth, outputChannels)
  val param = new TileGeneratorParameters(b, tileHeight, tileWidth, inputHeight, inputWidth, outputChannels)
  poke(dut.io.start, true)

  poke(dut.io.outputHCount, param.hCount)
  poke(dut.io.outputWCount, param.wCount)
  poke(dut.io.outputCCount, param.cOutCount)

  poke(dut.io.regularTileH, param.regularTileH)
  poke(dut.io.lastTileH, param.lastTileH)

  poke(dut.io.regularTileW, param.regularTileW)
  poke(dut.io.lastTileW, param.lastTileW)

  poke(dut.io.tileAccepted, false)
  for (tile <- ref.tileSeq) {
    while (peek(dut.io.tileValid) == 0) {
      step(1)
    }
    expect(dut.io.tileHeight, tile.height)
    expect(dut.io.tileWidth, tile.width)
    poke(dut.io.tileAccepted, true)
    step(1)
    poke(dut.io.tileAccepted, false)
  }
}

object A2fTileGeneratorTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val outputWidth = 10
    val outputHeight = 10
    val outputChannels = 2 * b
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val tileCountWidth = aAddrWidth

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    breakable {
      for ((tileHeight, tileWidth) <- List((4, 4), (5, 5), (10, 10))) {
        ok &= Driver.execute(driverArgs, () => new A2fTileGenerator(tileCountWidth))(
          dut => new A2fTileGeneratorTestSequence(dut, b, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels))
        if (!ok) {
          println(f"Failed for tileHeight:${tileHeight} tileWidth:${tileWidth}")
          break
        }
      }
    }
  }
}
