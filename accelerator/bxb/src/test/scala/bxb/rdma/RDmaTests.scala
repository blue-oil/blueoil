package bxb.rdma

import scala.collection._
import scala.math.{min}
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

import bxb.avalon.{WriteMasterIO}
import bxb.memory.{MemArray, WritePort}
import bxb.sync.{SemaphorePair}

class DummyRDmaRequest(request: DummyRequest, val startOfTile: Boolean) {
  var addr = request.addr
  var burst = request.burst
}

class DummyRDmaRequestSequenceGenerator(b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) {
  val tileGen = new DummyTileGenerator(b, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val requestSeq = mutable.ArrayBuffer[DummyRDmaRequest]()
  for (tile <- tileGen.tileSeq) {
    println(f"startAddress: ${tile.startAddress}, height: ${tile.height}, width: ${tile.width}, rowToRowDistance: ${tile.rowToRowDistance}")
    val requests = (new DummyWriteRequestGenerator(b, tile.height, tile.width, outputWidth, maxBurst, tile.startAddress)).requestSeq
    requestSeq += new DummyRDmaRequest(requests.head, startOfTile=true)
    requestSeq ++= requests.tail.map { new DummyRDmaRequest(_, startOfTile=false) }
  }
}

class RDmaTestRequestSequence(dut: RDma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyRDmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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
  poke(dut.io.rSync.rawZero, false)

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

class RDmaTestWaitRequest(dut: RDma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyRDmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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
  poke(dut.io.rSync.rawZero, false)

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

class RDmaTestRRawZero(dut: RDma, b: Int, avalonAddrWidth: Int, avalonDataWidth: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyRDmaRequestSequenceGenerator(b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)
  val param = new TileGeneratorParameters(b, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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
  poke(dut.io.rSync.rawZero, false)

  var acceptDelay = 1
  var pendingReads = 0
  breakable {
    for (req <- ref.requestSeq) {
      if (req.startOfTile) {
        expect(dut.io.avalonMaster.write, false)
        poke(dut.io.rSync.rawZero, true)
        for (_ <- 0 until acceptDelay) {
          step(1)
          expect(dut.io.avalonMaster.write, false)
        }
        acceptDelay = (acceptDelay + 2) % 5
        poke(dut.io.rSync.rawZero, false)
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

class RDmaTestModule(rmemSize: Int, avalonAddrWidth: Int, maxBurst: Int) extends Module {
  val rAddrWidth = Chisel.log2Up(rmemSize)
  val tileCountWidth = rAddrWidth
  val b = 32
  val avalonDataWidth = b * 2
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
    val outputSpace = Input(UInt(avalonAddrWidth.W))
    val rowDistance = Input(UInt(avalonAddrWidth.W))

    // Avalon interface
    val avalonMaster = WriteMasterIO(avalonAddrWidth, avalonDataWidth)

    // RMem interface
    val rmemWrite = Input(Vec(b, WritePort(rAddrWidth, 2)))

    // Synchronization interface
    val rWarZero = Output(Bool())
    val rWarDec = Input(Bool())
    val rRawInc = Input(Bool())
  })
  val rmem = Module(new MemArray(b, rmemSize, 2))
  rmem.io.write := io.rmemWrite
  val rSemaPair = Module(new SemaphorePair(2, 0, 2))
  io.rWarZero := rSemaPair.io.producer.warZero
  rSemaPair.io.producer.warDec := io.rWarDec
  rSemaPair.io.producer.rawInc := io.rRawInc
  val rdma = Module(new RDma(b, rAddrWidth, avalonAddrWidth, maxBurst))
  rdma.io.start := io.start
  rdma.io.outputAddress := io.outputAddress
  rdma.io.outputHCount := io.outputHCount
  rdma.io.outputWCount := io.outputWCount
  rdma.io.outputCCount := io.outputCCount
  rdma.io.regularTileH := io.regularTileH
  rdma.io.lastTileH := io.lastTileH
  rdma.io.regularTileW := io.regularTileW
  rdma.io.lastTileW := io.lastTileW
  rdma.io.regularRowToRowDistance := io.regularRowToRowDistance
  rdma.io.lastRowToRowDistance := io.lastRowToRowDistance
  rdma.io.outputSpace := io.outputSpace
  rdma.io.rowDistance := io.rowDistance
  io.avalonMaster <> rdma.io.avalonMaster
  rmem.io.read := rdma.io.rmemRead
  rdma.io.rmemQ := rmem.io.q
  rSemaPair.io.consumer <> rdma.io.rSync
}

class DdrAddress(val addr: Int, val startOfTile: Boolean, val endOfTile: Boolean) {
}

class RDmaTestRMemReading(dut: RDmaTestModule, rmemSize: Int, tileHeight: Int, tileWidth: Int, outputHeight: Int, outputWidth: Int, outputChannels: Int, maxBurst: Int, avalonDelays: List[Int]) extends PeekPokeTester(dut) {
  val expectedOutputMemory = Seq.fill(outputHeight * outputWidth * outputChannels * 2)(scala.util.Random.nextLong() & ((BigInt(0x1L) << 32) - 1))
  val avalonDataWidth = 32 * 2 // assume that bus width matches data size nicely
  val bytesPerElement = avalonDataWidth / 8
  
  val dataAddresses = mutable.ArrayBuffer[DdrAddress]()
  for (tileY <- 0 until outputHeight by tileHeight) {
    for (tileX <- 0 until outputWidth by tileWidth) {
      for (tileC <- 0 until outputChannels by 32) {
        val tileEndY = min(tileY + tileHeight, outputHeight)
        val tileEndX = min(tileX + tileWidth, outputWidth)
        for (y <- tileY until tileEndY) {
          for (x <- tileX until tileEndX) {
            val addr = (tileC / 32 * outputHeight * outputWidth + y * outputWidth + x) * bytesPerElement
            dataAddresses += new DdrAddress(addr, (y == tileY && x == tileX), (y == tileEndY - 1 && x == tileEndX - 1))
          }
        }
      }
    }
  }

  object RMemWriter {
    var lastAddressIdx = 0
    def done = (lastAddressIdx == dataAddresses.size)

    val rmemHalf = rmemSize / 2
    var rmemAddr = 0

    def pokeWrite(value: Boolean) = {
      for (row <- 0 until 32) {
        poke(dut.io.rmemWrite(row).enable, value)
      }
    }

    def pokeDataAt(addr: Int, dataAddr: Int) = {
      val valueLsbs = expectedOutputMemory(dataAddr >> 2)
      val valueMsbs = expectedOutputMemory((dataAddr >> 2) + 1)
      for (row <- 0 until 32) {
        var data = (((valueMsbs >> row) & 0x1) << 1) | ((valueLsbs >> row) & 0x1)
        poke(dut.io.rmemWrite(row).addr, addr)
        poke(dut.io.rmemWrite(row).data, data)
      }
    }

    def tryNext(): Unit = {
      if (done) {
        pokeWrite(false)
        poke(dut.io.rRawInc, false)
        poke(dut.io.rWarDec, false)
        return
      }
      val currentAddr = dataAddresses(lastAddressIdx)
      if (currentAddr.startOfTile && peek(dut.io.rWarZero) != 0) {
        pokeWrite(false)
        poke(dut.io.rRawInc, false)
        return
      }
      poke(dut.io.rWarDec, currentAddr.startOfTile)
      pokeWrite(true)
      pokeDataAt(rmemAddr, currentAddr.addr)
      poke(dut.io.rRawInc, currentAddr.endOfTile)
      lastAddressIdx += 1
      rmemAddr += 1
      if (currentAddr.endOfTile) {
        rmemAddr = if (rmemAddr < rmemHalf) rmemHalf else 0
      }
    }
  }

  val param = new TileGeneratorParameters(32, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst)

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
  poke(dut.io.avalonMaster.waitRequest, true)

  object AvalonChecker {
    var lastAddressIdx = 0
    var burstCountLeft = 0

    val avalonDelayQueue = mutable.Queue[Int]()
    avalonDelayQueue ++= avalonDelays
    def nextDelay() = {
      val first = avalonDelayQueue.dequeue
      avalonDelayQueue.enqueue(first)
      first
    }
    var acceptDelay = nextDelay()

    def done = (lastAddressIdx == dataAddresses.size)

    def tryNext(): Unit = {
      if (done) {
        return
      }
      val currentAddr = dataAddresses(lastAddressIdx)
      if (peek(dut.io.avalonMaster.write) == 0) {
        return
      }
      val addrLow = currentAddr.addr >> 2
      val addrHigh = addrLow + 1
      val expected = (expectedOutputMemory(addrHigh) << 32) | expectedOutputMemory(addrLow)
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
        acceptDelay = nextDelay()
      }
      if (burstCountLeft == 0) {
        expect(dut.io.avalonMaster.address, currentAddr.addr)
        burstCountLeft = peek(dut.io.avalonMaster.burstCount).toInt
      }
      lastAddressIdx += 1
      burstCountLeft -= 1
    }
  }

  while (!RMemWriter.done || !AvalonChecker.done) {
    RMemWriter.tryNext()
    AvalonChecker.tryNext()
    step(1)
  }
}

object RDmaTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val outputWidth = 64
    val outputHeight = 64
    val outputChannels = 2 * b
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val avalonAddrWidth = Chisel.log2Up(outputWidth * outputWidth * outputChannels * b * 2 / 8)
    val avalonDataWidth = b * 2

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    breakable {
      for (maxBurst <- List(1, 2, 4)) {
        for ((tileHeight, tileWidth) <- List((4, 4), (5, 5), (10, 10))) {
          println(f"running with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
          require(amemSize >= tileHeight * tileWidth)
          ok &= Driver.execute(driverArgs, () => new RDma(b, aAddrWidth, avalonAddrWidth, maxBurst))(
            dut => new RDmaTestRequestSequence(dut, b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new RDma(b, aAddrWidth, avalonAddrWidth, maxBurst))(
            dut => new RDmaTestWaitRequest(dut, b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst))
          ok &= Driver.execute(driverArgs, () => new RDma(b, aAddrWidth, avalonAddrWidth, maxBurst))(
            dut => new RDmaTestRRawZero(dut, b, avalonAddrWidth, avalonDataWidth, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst))
          for (delay <- 0 until 4) {
            ok &= Driver.execute(driverArgs, () => new RDmaTestModule(amemSize, avalonAddrWidth, maxBurst))(
              dut => new RDmaTestRMemReading(dut, amemSize, tileHeight, tileWidth, outputHeight, outputWidth, outputChannels, maxBurst, List(delay, 0, 0)))
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
