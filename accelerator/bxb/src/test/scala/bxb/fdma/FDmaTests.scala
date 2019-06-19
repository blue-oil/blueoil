package bxb.fdma

import scala.collection._
import scala.math.{min}
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

import bxb.avalon.{WriteMasterIO}
import bxb.memory.{TwoBlockMemArray, WritePort}
import bxb.sync.{SemaphorePair}

class DummyFDmaRequest(request: DummyRequest, val startOfTile: Boolean) {
  var addr = request.addr
  var burst = request.burst
}

class DummyFDmaRequestSequenceGenerator(b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, dataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) {
  val tileGen = new DummyTileGenerator(b, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val requestSeq = mutable.ArrayBuffer[DummyFDmaRequest]()
  for (tile <- tileGen.tileSeq) {
    println(f"startAddress: ${tile.startAddress}, height: ${tile.height}, width: ${tile.width}, rowToRowDistance: ${tile.rowToRowDistance}")
    val requests = (new DummyWriteRequestGenerator(avalonDataWidth,  dataWidth, tile.height, tile.width, outputWidth, maxBurst, tile.startAddress)).requestSeq
    requestSeq += new DummyFDmaRequest(requests.head, startOfTile=true)
    requestSeq ++= requests.tail.map { new DummyFDmaRequest(_, startOfTile=false) }
  }
}

class FDmaTestRequestSequence(dut: FDma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, dataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyFDmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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

  poke(dut.io.avalonMaster.waitRequest, false)
  poke(dut.io.fSync.rawZero, false)

  val timeout = t + 3 * param.hCount * param.wCount * param.cCount * b / maxBurst

  var pendingReads = 0
  breakable {
    for (req <- ref.requestSeq) {
      while (peek(dut.io.avalonMaster.write) == 0) {
        step(1)
        if (t >= timeout) {
          break
        }
      }
      for (i <- 0 until req.burst) {
        if (i == 0) {
          expect(dut.io.avalonMaster.address, req.addr)
        }
        expect(dut.io.avalonMaster.write, true)
        expect(dut.io.avalonMaster.burstCount, req.burst)
        step(1)
      }
    }
  }
}

class FDmaTestWaitRequest(dut: FDma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, dataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyFDmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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

  poke(dut.io.avalonMaster.waitRequest, false)
  poke(dut.io.fSync.rawZero, false)

  var acceptDelay = 1
  var pendingReads = 0
  breakable {
    for (req <- ref.requestSeq) {
      poke(dut.io.avalonMaster.waitRequest, true)
      while (peek(dut.io.avalonMaster.write) == 0) {
        step(1)
      }
      for (i <- 0 until req.burst) {
        poke(dut.io.avalonMaster.waitRequest, true)
        for (_ <- 0 until acceptDelay) {
          if (i == 0) {
            expect(dut.io.avalonMaster.address, req.addr)
          }
          expect(dut.io.avalonMaster.write, true)
          expect(dut.io.avalonMaster.burstCount, req.burst)
          step(1)
        }
        acceptDelay = (acceptDelay + 2) % 5
        poke(dut.io.avalonMaster.waitRequest, false)
        step(1)
      }
    }
  }
}

class FDmaTestFRawZero(dut: FDma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, dataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyFDmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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

  poke(dut.io.avalonMaster.waitRequest, false)
  poke(dut.io.fSync.rawZero, false)

  var acceptDelay = 1
  var pendingReads = 0
  breakable {
    for (req <- ref.requestSeq) {
      if (req.startOfTile) {
        expect(dut.io.avalonMaster.write, false)
        poke(dut.io.fSync.rawZero, true)
        for (_ <- 0 until acceptDelay) {
          step(1)
          expect(dut.io.avalonMaster.write, false)
        }
        acceptDelay = (acceptDelay + 2) % 5
        poke(dut.io.fSync.rawZero, false)
      }
      while (peek(dut.io.avalonMaster.write) == 0) {
        step(1)
      }
      for (i <- 0 until req.burst) {
        if (i == 0) {
          expect(dut.io.avalonMaster.address, req.addr)
        }
        expect(dut.io.avalonMaster.write, true)
        expect(dut.io.avalonMaster.burstCount, req.burst)
        step(1)
      }
    }
  }
}

class FDmaTestModule(fmemSize: Int, avalonAddrWidth: Int, maxBurst: Int) extends Module {
  val fAddrWidth = Chisel.log2Up(fmemSize)
  val tileCountWidth = fAddrWidth
  val b = 32
  val dataWidth = b * 16
  val avalonDataWidth = dataWidth / 4
  val io = IO(new Bundle {
    val start = Input(Bool())

    // Generator parameters
    val outputAddress = Input(UInt(avalonAddrWidth.W))
    val outputHCount = Input(UInt(6.W))
    val outputWCount = Input(UInt(6.W))
    val outputCCount = Input(UInt(6.W))
    val regularTileH = Input(UInt(tileCountWidth.W))
    val lastTileH = Input(UInt(tileCountWidth.W))
    val regularTileW = Input(UInt(tileCountWidth.W))
    val lastTileW = Input(UInt(tileCountWidth.W))
    val regularRowToRowDistance = Input(UInt(tileCountWidth.W))
    val lastRowToRowDistance = Input(UInt(tileCountWidth.W))
    val outputSpace = Input(UInt(tileCountWidth.W))
    val rowDistance = Input(UInt(avalonAddrWidth.W))

    // Avalon interface
    val avalonMaster = WriteMasterIO(avalonAddrWidth, avalonDataWidth)

    // FMem interface
    val fmemWrite = Input(Vec(b, WritePort(fAddrWidth, 16)))

    // Synchronization interface
    val fWarZero = Output(Bool())
    val fWarDec = Input(Bool())
    val fRawInc = Input(Bool())
  })
  val fmem = Module(new TwoBlockMemArray(b, fmemSize, 16))
  fmem.io.writeA := io.fmemWrite
  fmem.io.readA := 0.U.asTypeOf(fmem.io.readA)
  val fSemaPair = Module(new SemaphorePair(2, 0, 2))
  io.fWarZero := fSemaPair.io.producer.warZero
  fSemaPair.io.producer.warDec := io.fWarDec
  fSemaPair.io.producer.rawInc := io.fRawInc
  val fdma = Module(new FDma(b, fAddrWidth, avalonAddrWidth, avalonDataWidth, maxBurst))
  fdma.io.start := io.start
  fdma.io.outputAddress := io.outputAddress
  fdma.io.outputHCount := io.outputHCount
  fdma.io.outputWCount := io.outputWCount
  fdma.io.outputCCount := io.outputCCount
  fdma.io.regularTileH := io.regularTileH
  fdma.io.lastTileH := io.lastTileH
  fdma.io.regularTileW := io.regularTileW
  fdma.io.lastTileW := io.lastTileW
  fdma.io.regularRowToRowDistance := io.regularRowToRowDistance
  fdma.io.lastRowToRowDistance := io.lastRowToRowDistance
  fdma.io.outputSpace := io.outputSpace
  fdma.io.rowDistance := io.rowDistance
  io.avalonMaster <> fdma.io.avalonMaster
  fmem.io.readB := fdma.io.fmemRead
  fdma.io.fmemQ := fmem.io.qB
  fSemaPair.io.consumer <> fdma.io.fSync
}

class FDmaTestFMemReading(dut: FDmaTestModule, fmemSize: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int, avalonDelay: Int) extends PeekPokeTester(dut) {
  val itemWidth = 16
  val dataWidth = 32 * itemWidth
  val avalonDataWidth = dataWidth / 4

  val bytesPerWord = avalonDataWidth / 8
  val bytesPerElement = dataWidth / 8
  val wordsPerElement = dataWidth / avalonDataWidth
  val itemsPerWord = avalonDataWidth / itemWidth

  val randomSource = new scala.util.Random
  val expectedOutputMemory = Seq.fill(outputHeight * outputWidth * outputChannels * wordsPerElement)(BigInt(avalonDataWidth, randomSource))
  
  val dataAddresses = mutable.ArrayBuffer[DdrAddress]()
  breakable {
    for (tileY <- 0 until outputHeight by tileHeight) {
      for (tileX <- 0 until outputWidth by tileWidth) {
        for (tileC <- 0 until outputChannels by 32) {
          val tileEndY = min(tileY + tileHeight, outputHeight)
          val tileEndX = min(tileX + tileWidth, outputWidth)
          for (y <- tileY until tileEndY) {
            for (x <- tileX until tileEndX) {
              for (word <- 0 until wordsPerElement) {
                val addr = (tileC / 32 * outputHeight * outputWidth + y * outputWidth + x) * bytesPerElement + word * bytesPerWord
                val startOfTile = (y == tileY && x == tileX && word == 0)
                val endOfTile = (y == tileEndY - 1 && x == tileEndX - 1 && word == wordsPerElement - 1)
                dataAddresses += new DdrAddress(addr, startOfTile, endOfTile)
              }
            }
          }
          // if (tileC == 32)
          //   break
        }
      }
    }
  }

  class DdrAddress(val addr: Int, val startOfTile: Boolean, val endOfTile: Boolean) {
    val wordAddr = addr / bytesPerWord
  }

  object FMemWriter {
    private var lastAddressIdx = 0

    private val fmemHalf = fmemSize / 2
    private var fmemAddr = 0

    private val itemMask = (0x1 << itemWidth) - 1

    def done = (lastAddressIdx == dataAddresses.size)

    private def pokeWrite(value: Boolean) = {
      for (row <- 0 until 32) {
        poke(dut.io.fmemWrite(row).enable, value)
      }
    }

    def tryNext(): Unit = {
      if (done) {
        pokeWrite(false)
        poke(dut.io.fRawInc, false)
        poke(dut.io.fWarDec, false)
        return
      }
      for (rowStart <- 0 until 32 by itemsPerWord) {
        val currentAddr = dataAddresses(lastAddressIdx)
        if (currentAddr.startOfTile && peek(dut.io.fWarZero) != 0) {
          require(rowStart == 0)
          pokeWrite(false)
          poke(dut.io.fRawInc, false)
          return
        }
        if (currentAddr.startOfTile) {
          require(rowStart == 0)
          poke(dut.io.fWarDec, true)
        }
        val currentWord = expectedOutputMemory(currentAddr.wordAddr)
        for (i <- 0 until itemsPerWord) {
          val item = (currentWord >> (i * itemWidth)) & itemMask
          val row = rowStart + i
          poke(dut.io.fmemWrite(row).addr, fmemAddr)
          poke(dut.io.fmemWrite(row).data, item)
          poke(dut.io.fmemWrite(row).enable, true)
        }
        if (currentAddr.endOfTile) {
          poke(dut.io.fRawInc, true)
        }
        val lastWord = (rowStart == 32 - itemsPerWord)
        if (lastWord) {
          if (currentAddr.endOfTile) {
            fmemAddr = if (fmemAddr < fmemHalf) fmemHalf else 0
          }
          else {
            fmemAddr += 1
          }
        }
        lastAddressIdx += 1
      }
    }
  }

  val param = new TileGeneratorParameters(32, avalonDataWidth, dataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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
  poke(dut.io.avalonMaster.waitRequest, false)

  object AvalonChecker {
    private var lastAddressIdx = 0
    private var burstCountLeft = 0
    private var acceptDelay = avalonDelay

    def done = (lastAddressIdx == dataAddresses.size)

    def tryNext(): Unit = {
      if (done) {
        return
      }
      val currentAddr = dataAddresses(lastAddressIdx)
      if (peek(dut.io.avalonMaster.write) == 0) {
        return
      }
      val expected = expectedOutputMemory(currentAddr.wordAddr)
      expect(dut.io.avalonMaster.writeData, expected)
      if (acceptDelay != 0) {
        poke(dut.io.avalonMaster.waitRequest, true)
        if (burstCountLeft == 0) {
          expect(dut.io.avalonMaster.address, currentAddr.addr)
        }
        acceptDelay -= 1
        return
      }
      else {
        poke(dut.io.avalonMaster.waitRequest, false)
        acceptDelay = avalonDelay
      }
      if (burstCountLeft == 0) {
        expect(dut.io.avalonMaster.address, currentAddr.addr)
        burstCountLeft = peek(dut.io.avalonMaster.burstCount).toInt
      }
      lastAddressIdx += 1
      burstCountLeft -= 1
    }
  }

  while (!FMemWriter.done || !AvalonChecker.done) {
    FMemWriter.tryNext()
    AvalonChecker.tryNext()
    step(1)
  }
}

object FDmaTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val outputWidth = 20
    val outputHeight = 20
    val outputChannels = b * 2
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val avalonAddrWidth = Chisel.log2Up(outputWidth * outputWidth * outputChannels * b * 16 / 8)

    val depthDataWidth = b * 16
    val avalonDataWidth = depthDataWidth / 4

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    breakable {
      for (maxBurst <- List(4)) {
        for ((tileHeight, tileWidth) <- List((4, 4), (5, 5), (10, 10), (outputHeight, outputWidth))) {
          println(f"running with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
          require(amemSize >= tileHeight * tileWidth)
          ok &= Driver.execute(driverArgs, () => new FDma(b, aAddrWidth, avalonAddrWidth, avalonDataWidth, maxBurst))(
            dut => new FDmaTestRequestSequence(dut, b, avalonAddrWidth, avalonDataWidth, depthDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new FDma(b, aAddrWidth, avalonAddrWidth, avalonDataWidth, maxBurst))(
            dut => new FDmaTestWaitRequest(dut, b, avalonAddrWidth, avalonDataWidth, depthDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new FDma(b, aAddrWidth, avalonAddrWidth, avalonDataWidth, maxBurst))(
            dut => new FDmaTestFRawZero(dut, b, avalonAddrWidth, avalonDataWidth, depthDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst))
          for (delay <- 0 until 4) {
            ok &= Driver.execute(driverArgs, () => new FDmaTestModule(amemSize, avalonAddrWidth, maxBurst))(
              dut => new FDmaTestFMemReading(dut, amemSize, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst, delay))
            if (!ok) {
              println(f"Failed for maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth} delay ${delay}")
              break
            }
          }
          if (!ok) {
            println(f"Failed for maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
            break
          }
        }
      }
    }
  }
}
