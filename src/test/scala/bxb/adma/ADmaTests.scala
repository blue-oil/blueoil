package bxb.adma

import scala.collection._
import scala.math.{min}
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

import bxb.memory.{MemArray, ReadPort}
import bxb.sync.{SemaphorePair}

class DummyADmaRequest(request: DummyRequest, val startOfTile: Boolean) {
  var addr = request.addr
  var burst = request.burst
}

class DummyADmaRequestSequenceGenerator(b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) {
  val tileGen = new DummyTileGenerator(b, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val requestSeq = mutable.ArrayBuffer[DummyADmaRequest]()
  for (tile <- tileGen.tileSeq) {
    println(f"startAddress: ${tile.startAddress}, height: ${tile.height}, width: ${tile.width}, rowToRowDistance: ${tile.rowToRowDistance}")
    val requests = (new DummyRequestGenerator(b, tile.height, tile.width, inputWidth, maxBurst, tile.startAddress)).requestSeq
    requestSeq += new DummyADmaRequest(requests.head, startOfTile=true)
    requestSeq ++= requests.tail.map { new DummyADmaRequest(_, startOfTile=false) }
  }
}

class ADmaTestRequestSequence(dut: ADma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyADmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
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

  poke(dut.io.topBottomLeftPad, param.topBottomLeftPad)
  poke(dut.io.topBottomMiddlePad, param.topBottomMiddlePad)
  poke(dut.io.topBottomRightPad, param.topBottomRightPad)
  poke(dut.io.sidePad, param.sidePad)


  poke(dut.io.avalonMasterWaitRequest, false)
  poke(dut.io.aWarZero, false)

  val timeout = t + 3 * param.hCount * param.wCount * param.cCount * b / maxBurst

  var pendingReads = 0
  breakable {
    for (req <- ref.requestSeq) {
      while (peek(dut.io.avalonMasterRead) == 0 && t < timeout) {
        poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
        if (pendingReads > 0) {
          pendingReads -= 1
        }
        step(1)
      }
      poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
      if (pendingReads > 0) {
        pendingReads -= 1
      }
      if (t == timeout) {
        break
      }
      expect(dut.io.avalonMasterAddress, req.addr)
      expect(dut.io.avalonMasterBurstCount, req.burst)
      pendingReads += peek(dut.io.avalonMasterBurstCount).toInt
      step(1)
    }
  }
}

class ADmaTestWaitRequest(dut: ADma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyADmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
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

  poke(dut.io.topBottomLeftPad, param.topBottomLeftPad)
  poke(dut.io.topBottomMiddlePad, param.topBottomMiddlePad)
  poke(dut.io.topBottomRightPad, param.topBottomRightPad)
  poke(dut.io.sidePad, param.sidePad)

  poke(dut.io.aWarZero, false)

  var acceptDelay = 1
  var pendingReads = 0
  for (req <- ref.requestSeq) {
    poke(dut.io.avalonMasterWaitRequest, true)
    while (peek(dut.io.avalonMasterRead) == 0) {
      poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
      if (pendingReads > 0) {
        println(f"${t}: pendingReads X ${pendingReads}")
        pendingReads -= 1
      }
      step(1)
    }
    poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
    if (pendingReads > 0) {
      println(f"${t}: pendingReads Y ${pendingReads}")
      pendingReads -= 1
    }
    for (_ <- 0 until acceptDelay) {
      step(1)
      poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
      if (pendingReads > 0) {
        println(f"${t}: pendingReads Z ${pendingReads}")
        pendingReads -= 1
      }
      expect(dut.io.avalonMasterAddress, req.addr)
      expect(dut.io.avalonMasterRead, true)
      expect(dut.io.avalonMasterBurstCount, req.burst)
    }
    acceptDelay = (acceptDelay + 2) % 5
    poke(dut.io.avalonMasterWaitRequest, false)
    pendingReads += peek(dut.io.avalonMasterBurstCount).toInt
    step(1)
  }
}

class ADmaTestAWarZero(dut: ADma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyADmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
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

  poke(dut.io.topBottomLeftPad, param.topBottomLeftPad)
  poke(dut.io.topBottomMiddlePad, param.topBottomMiddlePad)
  poke(dut.io.topBottomRightPad, param.topBottomRightPad)
  poke(dut.io.sidePad, param.sidePad)

  poke(dut.io.avalonMasterWaitRequest, false)

  var acceptDelay = 1
  var pendingReads = 0
  for (req <- ref.requestSeq) {
    if (req.startOfTile) {
      poke(dut.io.aWarZero, true)
      for (_ <- 0 until acceptDelay) {
        poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
        if (pendingReads > 0) {
          pendingReads -= 1
        }
        step(1)
        expect(dut.io.avalonMasterRead, false)
      }
      poke(dut.io.aWarZero, false)
    }
    while (peek(dut.io.avalonMasterRead) == 0) {
      poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
      if (pendingReads > 0) {
        pendingReads -= 1
      }
      step(1)
    }
    poke(dut.io.avalonMasterReadDataValid, pendingReads > 0)
    if (pendingReads > 0) {
      pendingReads -= 1
    }
    expect(dut.io.avalonMasterAddress, req.addr)
    expect(dut.io.avalonMasterBurstCount, req.burst)
    pendingReads += peek(dut.io.avalonMasterBurstCount).toInt
    step(1)
  }
}

class ADmaTestModule(amemSize: Int, avalonAddrWidth: Int, maxBurst: Int) extends Module {
  val aAddrWidth = Chisel.log2Up(amemSize)
  val tileCountWidth = aAddrWidth
  val b = 32
  val avalonDataWidth = b * 2
  val io = IO(new Bundle {
    val start = Input(Bool())

    // Tile generation parameters
    val inputAddress = Input(UInt(avalonAddrWidth.W))
    val inputHCount = Input(UInt(6.W))
    val inputWCount = Input(UInt(6.W))
    val inputCCount = Input(UInt(6.W))

    val topTileH = Input(UInt(tileCountWidth.W))
    val middleTileH = Input(UInt(tileCountWidth.W))
    val bottomTileH = Input(UInt(tileCountWidth.W))

    val leftTileW = Input(UInt(tileCountWidth.W))
    val middleTileW = Input(UInt(tileCountWidth.W))
    val rightTileW = Input(UInt(tileCountWidth.W))

    val leftRowToRowDistance = Input(UInt(tileCountWidth.W))
    val middleRowToRowDistance = Input(UInt(tileCountWidth.W))
    val rightRowToRowDistance = Input(UInt(tileCountWidth.W))

    val leftStep = Input(UInt(avalonAddrWidth.W))
    val middleStep = Input(UInt(avalonAddrWidth.W))

    val topRowDistance = Input(UInt(avalonAddrWidth.W))
    val midRowDistance = Input(UInt(avalonAddrWidth.W))

    val inputSpace = Input(UInt(avalonAddrWidth.W))

    val topBottomLeftPad = Input(UInt(tileCountWidth.W))
    val topBottomMiddlePad = Input(UInt(tileCountWidth.W))
    val topBottomRightPad = Input(UInt(tileCountWidth.W))
    val sidePad = Input(UInt(tileCountWidth.W))

    // Avalon test interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterRead = Output(Bool())
    val avalonMasterBurstCount = Output(UInt(10.W))
    val avalonMasterWaitRequest = Input(Bool())

    val avalonMasterReadDataValid = Input(Bool())
    val avalonMasterReadData = Input(UInt(avalonDataWidth.W))

    // AMem test interface
    val amemRead = Input(Vec(b, ReadPort(aAddrWidth)))
    val amemQ = Output(Vec(b, UInt(2.W)))

    // AMem sync test interface
    val aRawZero = Output(Bool())
    val aRawDec = Input(Bool())
    val aWarInc = Input(Bool())
  })
  val amem = Module(new MemArray(b, amemSize, 2))
  amem.io.read := io.amemRead
  io.amemQ := amem.io.q
  val aSemaPair = Module(new SemaphorePair(2, 0, 2))
  io.aRawZero := aSemaPair.io.consumer.rawZero
  aSemaPair.io.consumer.rawDec := io.aRawDec
  aSemaPair.io.consumer.warInc := io.aWarInc
  val adma = Module(new ADma(b, aAddrWidth, avalonAddrWidth, maxBurst))
  adma.io.start := io.start
  adma.io.inputAddress := io.inputAddress
  adma.io.inputHCount := io.inputHCount
  adma.io.inputWCount := io.inputWCount
  adma.io.inputCCount := io.inputCCount
  adma.io.topTileH := io.topTileH
  adma.io.middleTileH := io.middleTileH
  adma.io.bottomTileH := io.bottomTileH
  adma.io.leftTileW := io.leftTileW
  adma.io.middleTileW := io.middleTileW
  adma.io.rightTileW := io.rightTileW
  adma.io.leftRowToRowDistance := io.leftRowToRowDistance
  adma.io.middleRowToRowDistance := io.middleRowToRowDistance
  adma.io.rightRowToRowDistance := io.rightRowToRowDistance
  adma.io.leftStep := io.leftStep
  adma.io.middleStep := io.middleStep
  adma.io.topRowDistance := io.topRowDistance
  adma.io.midRowDistance := io.midRowDistance
  adma.io.inputSpace := io.inputSpace
  adma.io.topBottomLeftPad := io.topBottomLeftPad
  adma.io.topBottomMiddlePad := io.topBottomMiddlePad
  adma.io.topBottomRightPad := io.topBottomRightPad
  adma.io.sidePad := io.sidePad
  io.avalonMasterAddress := adma.io.avalonMasterAddress
  io.avalonMasterRead := adma.io.avalonMasterRead
  io.avalonMasterBurstCount := adma.io.avalonMasterBurstCount
  adma.io.avalonMasterWaitRequest := io.avalonMasterWaitRequest
  adma.io.avalonMasterReadDataValid := io.avalonMasterReadDataValid
  adma.io.avalonMasterReadData := io.avalonMasterReadData
  amem.io.write := adma.io.amemWrite
  adma.io.aWarZero := aSemaPair.io.producer.warZero
  aSemaPair.io.producer.warDec := adma.io.aWarDec
  aSemaPair.io.producer.rawInc := adma.io.aRawInc
}

class ADmaTestAMemWriting(dut: ADmaTestModule, amemSize: Int, tileHeight: Int, tileWidth: Int, inputHeight: Int, inputWidth: Int, inputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  // input as 32 bit words laying in DDR
  val inputMemory = Seq.fill(inputHeight * inputWidth * inputChannels * 2)(scala.util.Random.nextLong() & ((0x1L << 32) - 1))
  val avalonDataWidth = 32 * 2 / 8 // assume that bus width matches data size nicely

  object AvalonStub {
    val requests = mutable.Queue[DummyRequest]()

    def input64(addr: Int): Long = {
      // XXX: address supposed to point to 64 bit value and be aligned
      // (we indirectyly check it as part of address generation tests)
      val wordLow = addr >> 2
      val wordHigh = wordLow + 1
      return (inputMemory(wordHigh) << 32) | inputMemory(wordLow)
    }

    def next() {
      // serve pending requests
      poke(dut.io.avalonMasterWaitRequest, false)
      if (requests.isEmpty) {
        poke(dut.io.avalonMasterReadDataValid, false)
      }
      else {
        val req = requests.front
        poke(dut.io.avalonMasterReadDataValid, true)
        poke(dut.io.avalonMasterReadData, input64(req.addr))
        req.burst -= 1
        req.addr += avalonDataWidth
        if (req.burst == 0)
          requests.dequeue()
      }
      // queue new requests if any
      if (peek(dut.io.avalonMasterRead).toInt == 1) {
        requests.enqueue(new DummyRequest(peek(dut.io.avalonMasterAddress).toInt, peek(dut.io.avalonMasterBurstCount).toInt))
      }
    }
  }

  def doStep() = {
    AvalonStub.next()
    step(1)
  }

  val ref = new DummyTileGenerator(32, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)
  val param = new TileGeneratorParameters(32, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst)

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

  poke(dut.io.topBottomLeftPad, param.topBottomLeftPad)
  poke(dut.io.topBottomMiddlePad, param.topBottomMiddlePad)
  poke(dut.io.topBottomRightPad, param.topBottomRightPad)
  poke(dut.io.sidePad, param.sidePad)

  poke(dut.io.avalonMasterWaitRequest, false)

  poke(dut.io.aWarInc, false)
  poke(dut.io.aRawDec, false)

  var amemAddr = 0
  val amemHalf = amemSize / 2
  for (tileY <- -param.pad until (inputHeight) by (tileHeight - param.dep)) {
    for (tileX <- -param.pad until (inputWidth) by (tileWidth - param.dep)) {
      for (tileC <- 0 until inputChannels by 32) {
        val dataY = if (tileY < 0) 0 else tileY
        val dataEndY = if (tileY + tileHeight > inputHeight) inputHeight else tileY + tileHeight
        val dataHeight = dataEndY - dataY

        val dataX = if (tileX < 0) 0 else tileX
        val dataEndX = if (tileX + tileWidth > inputWidth) inputWidth else tileX + tileWidth
        val dataWidth = dataEndX - dataX

        if (dataWidth + param.pad > param.dep && dataHeight + param.pad > param.dep) {
          while (peek(dut.io.aRawZero).toInt != 0) {
            doStep()
            poke(dut.io.aWarInc, false)
          }
          poke(dut.io.aRawDec, true)
          for (y <- tileY until min(tileY + tileHeight, inputHeight + param.pad)) {
            for (x <- tileX until min(tileX + tileWidth, inputWidth + param.pad)) {
              val inputAddr = tileC / 32 * inputHeight * inputWidth + y * inputWidth + x
              for (row <- 0 until 32) {
                poke(dut.io.amemRead(row).addr, amemAddr)
                poke(dut.io.amemRead(row).enable, true)
              }
              doStep()
              poke(dut.io.aWarInc, false)
              poke(dut.io.aRawDec, false)
              val isPad = !(y >= 0 && y < inputHeight && x >= 0 && x < inputWidth)
              val inputLow = if (isPad) 0 else inputMemory(inputAddr * 2)
              val inputHigh = if (isPad) 0 else inputMemory(inputAddr * 2 + 1)
              for (row <- 0 until 32) {
                var expected = (((inputHigh >> row) & 0x1) << 1) | ((inputLow >> row) & 0x1)
                expect(dut.io.amemQ(row), expected)
              }
              amemAddr = amemAddr + 1
            }
          }
          // ping pong buffering assumed
          amemAddr = if (amemAddr < amemHalf) amemHalf else 0
          poke(dut.io.aWarInc, true)
        }
      }
    }
  }
}

object ADmaTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val inputWidth = 20
    val inputHeight = 20
    val inputChannels = 2 * b
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val avalonAddrWidth = Chisel.log2Up(inputWidth * inputWidth * inputChannels * b * 2 / 8)
    val avalonDataWidth = b * 2

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    breakable {
      for (maxBurst <- List(1, 2, 4)) {
        for ((tileHeight, tileWidth) <- List((4, 4), (5, 5), (10, 10))) {
          println(f"running with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
          require(amemSize >= tileHeight * tileWidth)
          ok &= Driver.execute(driverArgs, () => new ADma(b, aAddrWidth, avalonAddrWidth, maxBurst))(
            dut => new ADmaTestRequestSequence(dut, b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new ADma(b, aAddrWidth, avalonAddrWidth, maxBurst))(
            dut => new ADmaTestWaitRequest(dut, b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new ADma(b, aAddrWidth, avalonAddrWidth, maxBurst))(
            dut => new ADmaTestAWarZero(dut, b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new ADmaTestModule(amemSize, avalonAddrWidth, maxBurst))(
            dut => new ADmaTestAMemWriting(dut, amemSize, tileHeight, tileWidth, inputHeight, inputWidth, inputChannels, maxBurst))
          if (!ok) {
            println(f"Failed for maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
            break
          }
        }
      }
    }
  }
}
