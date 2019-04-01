package bxb

import scala.collection._
import scala.math.{min}

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class Reference(b: Int, inputHeight: Int, inputWidth: Int, tileHeight: Int, tileWidth: Int) {
  val inputChannels = b

  val outputHeight = inputHeight
  val outputWidth = inputWidth
  val outputChannels = b

  val kernelHeight = 3
  val kernelWidth = 3

  val pad = 1
  val dep = 2

  val maxBurst = 32

  val admaAvalonDataWidth = b * 2
  val admaAvalonDataByteWidth = admaAvalonDataWidth / 8
  val wdmaAvalonDataWidth = b * 1
  val fdmaAvalonDataWidth = 128
  val fdmaDataWidth = b * 16

  val inputTileHeight = tileHeight
  val inputTileWidth = tileWidth

  val outputTileHeight = inputTileHeight - dep
  val outputTileWidth = inputTileWidth - dep

  val admaParam = new bxb.adma.TileGeneratorParameters(b, admaAvalonDataWidth, inputTileHeight, inputTileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val fdmaParam = new bxb.fdma.TileGeneratorParameters(b, fdmaAvalonDataWidth,  fdmaDataWidth, outputTileHeight, outputTileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val wdmaParam = new {
    val hCount = fdmaParam.hCount
    val wCount = fdmaParam.wCount
    val blockCount = (outputChannels / b * inputChannels / b * kernelHeight * kernelWidth)
  }
  val a2fParam = new bxb.a2f.TileGeneratorParameters(outputTileHeight, outputTileWidth, outputHeight, outputWidth) {
    val cCount = inputChannels / b
    val kernelVCount = kernelHeight
    val kernelHCount = kernelWidth
    val step = 1
    val gap = 3
  }

  val input = Seq.fill(inputHeight, inputWidth, inputChannels)(scala.util.Random.nextInt(4))

  val kernels = Seq.fill(kernelHeight, kernelWidth, outputChannels, inputChannels)(scala.util.Random.nextInt(2))
  val decodedKernels = kernels.map(_.map(_.map(_.map({x => if (x == 0) 1 else -1}))))

  val output = mutable.Seq.fill(outputHeight, outputWidth, outputChannels)(0)

  // Convolution
  for (iy <- -pad until (inputHeight + pad)) {
    for (ix <- -pad until (inputWidth + pad)) {
      for (oc <- 0 until outputChannels) {
        var pixel = 0
        // iterating over input sector convolved
        // to get output pixel
        for (ky <- 0 until 3) {
          for (kx <- 0 until 3) {
            for (kc <- 0 until inputChannels) {
              val y = iy + ky
              val x = ix + kx
              if (y >= 0 && y < inputHeight && x >= 0 && x < inputWidth) {
                pixel += input(iy + ky)(ix + kx)(kc) * decodedKernels(ky)(kx)(oc)(kc)
              }
            }
          }
        }
        val oy = iy + pad
        val ox = ix + pad
        if (oy < outputHeight && ox < outputWidth) {
          output(oy)(ox)(oc) = pixel
        }
      }
    }
  }

  // Input sequence
  val inputAddresses = mutable.ArrayBuffer[Int]()
  for (tileY <- -pad until (inputHeight) by (inputTileHeight - dep)) {
    for (tileX <- -pad until (inputWidth) by (inputTileWidth - dep)) {
      for (tileC <- 0 until inputChannels by b) {
        val dataY = if (tileY < 0) 0 else tileY
        val dataEndY = if (tileY + inputTileHeight > inputHeight) inputHeight else tileY + inputTileHeight
        val dataHeight = dataEndY - dataY

        val dataX = if (tileX < 0) 0 else tileX
        val dataEndX = if (tileX + inputTileWidth > inputWidth) inputWidth else tileX + inputTileWidth
        val dataWidth = dataEndX - dataX

        if (dataWidth + pad > dep && dataHeight + pad > dep) {
          for (y <- tileY until min(tileY + inputTileHeight, inputHeight + pad)) {
            for (x <- tileX until min(tileX + inputTileWidth, inputWidth + pad)) {
              if (y >= 0 && y < inputHeight && x >= 0 && x < inputWidth) {
                val inputAddr = (tileC / b * inputHeight * inputWidth + y * inputWidth + x) * admaAvalonDataByteWidth
                inputAddresses += inputAddr
              }
            }
          }
        }
      }
    }
  }

  // Weight sequence

  // Output sequence
  val bytesPerElement = b * 16 / 8
  val bytesPerWord = 128 / 8
  val wordsPerElement = bytesPerElement / bytesPerWord
  val outputAddresses = mutable.ArrayBuffer[Int]()
  for (tileY <- 0 until outputHeight by outputTileHeight) {
    for (tileX <- 0 until outputWidth by outputTileWidth) {
      for (tileC <- 0 until outputChannels by b) {
        val tileEndY = min(tileY + outputTileHeight, outputHeight)
        val tileEndX = min(tileX + outputTileWidth, outputWidth)
        for (y <- tileY until tileEndY) {
          for (x <- tileX until tileEndX) {
            for (word <- 0 until wordsPerElement) {
              val outputAddr = (tileC / b * outputHeight * outputWidth + y * outputWidth + x) * bytesPerElement + word * bytesPerWord
              outputAddresses += outputAddr
            }
          }
        }
      }
    }
  }
}

class BxbTests(dut: Bxb, b: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  val ref = new Reference(b, inputHeight, inputWidth, tileHeight, tileWidth)

  class Request(var addr: Int, var burst: Int) {
  }

  object ADmaAvalonStub {
    val requests = mutable.Queue[Request]()
    var lastAddrIdx = 0
    var nextAddrIdx = 0
    def done = (lastAddrIdx == ref.inputAddresses.size)
    def tryNext(): Unit = {
      poke(dut.io.admaAvalonWaitRequest, false)
      if (done) {
        poke(dut.io.admaAvalonReadDataValid, false)
        return
      }
      if (requests.isEmpty) {
        poke(dut.io.admaAvalonReadDataValid, false)
      }
      else {
        val req = requests.front
        poke(dut.io.admaAvalonReadDataValid, true)
        // TODO: feed the data
        req.burst -= 1
        req.addr += ref.admaAvalonDataWidth / 8
        if (req.burst == 0)
          requests.dequeue()
        lastAddrIdx += 1
      }
      if (peek(dut.io.admaAvalonRead).toInt == 1) {
        expect(dut.io.admaAvalonAddress, ref.inputAddresses(nextAddrIdx))
        requests.enqueue(new Request(peek(dut.io.admaAvalonAddress).toInt, peek(dut.io.admaAvalonBurstCount).toInt))
        nextAddrIdx += peek(dut.io.admaAvalonBurstCount).toInt
      }
    }
  }

  object WDmaAvalonStub {
    val requests = mutable.Queue[Request]()
    def tryNext(): Unit = {
      poke(dut.io.wdmaAvalonWaitRequest, false)
      if (requests.isEmpty) {
        poke(dut.io.wdmaAvalonReadDataValid, false)
      }
      else {
        val req = requests.front
        poke(dut.io.wdmaAvalonReadDataValid, true)
        // TODO: feed the data
        req.burst -= 1
        req.addr += ref.wdmaAvalonDataWidth / 8
        if (req.burst == 0)
          requests.dequeue()
      }
      if (peek(dut.io.wdmaAvalonRead).toInt == 1) {
        requests.enqueue(new Request(peek(dut.io.wdmaAvalonAddress).toInt, peek(dut.io.wdmaAvalonBurstCount).toInt))
      }
    }
  }

  object FDmaAvalonStub {
    val dataAddresses = ref.outputAddresses

    val requests = mutable.Queue[Request]()
    var lastAddrIdx = 0

    var burstCountLeft = 0

    def done = (lastAddrIdx == dataAddresses.size)

    def tryNext(): Unit = {
      if (done) {
        return
      }
      val currentAddr = dataAddresses(lastAddrIdx)
      if (peek(dut.io.fdmaAvalonWrite) == 0) {
        return
      }
      poke(dut.io.fdmaAvalonWaitRequest, false)
      if (burstCountLeft == 0) {
        expect(dut.io.fdmaAvalonAddress, ref.outputAddresses(lastAddrIdx))
        burstCountLeft = peek(dut.io.fdmaAvalonBurstCount).toInt
      }
      lastAddrIdx += 1
      burstCountLeft -= 1
      // TODO: check the data
    }
  }

  def writeCsr(field: Int, value: Int) = {
    poke(dut.io.csrSlaveAddress, field)
    poke(dut.io.csrSlaveWriteData, value)
    poke(dut.io.csrSlaveWrite, true)
  }

  val parameters = List(
    (BxbCsrField.admaInputAddress, 0),
    (BxbCsrField.admaInputHCount, ref.admaParam.hCount),
    (BxbCsrField.admaInputWCount, ref.admaParam.wCount),
    (BxbCsrField.admaInputCCount, ref.admaParam.cCount),
    (BxbCsrField.admaTopTileH, ref.admaParam.topTileH),
    (BxbCsrField.admaMiddleTileH, ref.admaParam.middleTileH),
    (BxbCsrField.admaBottomTileH, ref.admaParam.bottomTileH),
    (BxbCsrField.admaLeftTileW, ref.admaParam.leftTileW),
    (BxbCsrField.admaMiddleTileW, ref.admaParam.middleTileW),
    (BxbCsrField.admaRightTileW, ref.admaParam.rightTileW),
    (BxbCsrField.admaLeftRowToRowDistance, ref.admaParam.leftRowToRowDistance),
    (BxbCsrField.admaMiddleRowToRowDistance, ref.admaParam.middleRowToRowDistance),
    (BxbCsrField.admaRightRowToRowDistance, ref.admaParam.rightRowToRowDistance),
    (BxbCsrField.admaLeftStep, ref.admaParam.leftStep),
    (BxbCsrField.admaMiddleStep, ref.admaParam.middleStep),
    (BxbCsrField.admaTopRowDistance, ref.admaParam.topRowDistance),
    (BxbCsrField.admaMidRowDistance, ref.admaParam.midRowDistance),
    (BxbCsrField.admaInputSpace, ref.admaParam.inputSpace),
    (BxbCsrField.admaTopBottomLeftPad, ref.admaParam.topBottomLeftPad),
    (BxbCsrField.admaTopBottomMiddlePad, ref.admaParam.topBottomMiddlePad),
    (BxbCsrField.admaTopBottomRightPad, ref.admaParam.topBottomRightPad),
    (BxbCsrField.admaSidePad, ref.admaParam.sidePad),

    (BxbCsrField.wdmaStartAddress, 0),
    (BxbCsrField.wdmaOutputHCount, ref.wdmaParam.hCount),
    (BxbCsrField.wdmaOutputWCount, ref.wdmaParam.wCount),
    (BxbCsrField.wdmaKernelBlockCount, ref.wdmaParam.blockCount),

    (BxbCsrField.fdmaOutputAddress, 0),
    (BxbCsrField.fdmaOutputHCount, ref.fdmaParam.hCount),
    (BxbCsrField.fdmaOutputWCount, ref.fdmaParam.wCount),
    (BxbCsrField.fdmaOutputCCount, ref.fdmaParam.cCount),
    (BxbCsrField.fdmaRegularTileH, ref.fdmaParam.regularTileH),
    (BxbCsrField.fdmaLastTileH, ref.fdmaParam.lastTileH),
    (BxbCsrField.fdmaRegularTileW, ref.fdmaParam.regularTileW),
    (BxbCsrField.fdmaLastTileW, ref.fdmaParam.lastTileW),
    (BxbCsrField.fdmaRegularRowToRowDistance, ref.fdmaParam.regularRowToRowDistance),
    (BxbCsrField.fdmaLastRowToRowDistance, ref.fdmaParam.lastRowToRowDistance),
    (BxbCsrField.fdmaOutputSpace, ref.fdmaParam.outputSpace),
    (BxbCsrField.fdmaRowDistance, ref.fdmaParam.rowDistance),

    (BxbCsrField.a2fInputCCount, ref.a2fParam.cCount),
    (BxbCsrField.a2fKernelVCount, ref.a2fParam.kernelVCount),
    (BxbCsrField.a2fKernelHCount, ref.a2fParam.kernelHCount),
    (BxbCsrField.a2fTileStep, ref.a2fParam.step),
    (BxbCsrField.a2fTileGap, ref.a2fParam.gap),
    (BxbCsrField.a2fOutputHCount, ref.a2fParam.hCount),
    (BxbCsrField.a2fOutputWCount, ref.a2fParam.wCount),
    (BxbCsrField.a2fRegularTileH, ref.a2fParam.regularTileH),
    (BxbCsrField.a2fLastTileH, ref.a2fParam.lastTileH),
    (BxbCsrField.a2fRegularTileW, ref.a2fParam.regularTileW),
    (BxbCsrField.a2fLastTileW, ref.a2fParam.lastTileW),
    (BxbCsrField.start, 1)
  )

  for ((field, value) <- parameters) {
    ADmaAvalonStub.tryNext()
    WDmaAvalonStub.tryNext()
    FDmaAvalonStub.tryNext()
    writeCsr(field, value)
    step(1)
  }

  def tryNext() = {
    ADmaAvalonStub.tryNext()
    WDmaAvalonStub.tryNext()
    FDmaAvalonStub.tryNext()
    step(1)
  }
  poke(dut.io.csrSlaveWrite, false)
  poke(dut.io.csrSlaveAddress, BxbCsrField.statusRegister)
  tryNext()
  while (!ADmaAvalonStub.done || !FDmaAvalonStub.done) {
    tryNext()
  }
  tryNext()
  tryNext()
  expect(dut.io.csrSlaveReadData, 15)
}

object BxbTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val inputHeight = 10
    val inputWidth = 10
    val inputChannels = 1 * b
    // val inputHeight = 100
    // val inputWidth = 100

    val tileHeight = 4
    val tileWidth = 4
    // val tileHeight = 32
    // val tileWidth = 32

    val dataMemSize = 4096
    val wmemSize = 1024

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    println(f"running with tileHeight:${tileHeight} tileWidth:${tileWidth}")
    require(dataMemSize >= tileHeight * tileWidth)
    ok &= Driver.execute(driverArgs, () => new Bxb(dataMemSize, wmemSize))(dut => new BxbTests(dut, b, inputHeight, inputWidth, inputChannels, tileHeight, tileWidth))
  }
}
