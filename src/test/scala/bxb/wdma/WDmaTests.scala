package bxb.wdma

import scala.collection._
import scala.math.{min}
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

import bxb.memory.{PackedBlockRam, ReadPort}
import bxb.sync.{SemaphorePair}

class WDmaTestModule(b: Int, avalonDataWidth: Int, wmemSize: Int) extends Module {
  val avalonAddrWidth = 32
  val wAddrWidth = Chisel.log2Up(wmemSize)
  require(Chisel.isPow2(wmemSize))
  require(wmemSize % b == 0)
  // FIXME: rid of copypaste
  val hCountWidth = 6
  val wCountWidth = 6
  val blockCountWidth = 14
  val io = IO(new Bundle {
    val start = Input(Bool())

    val startAddress = Input(UInt(avalonAddrWidth.W))
    // - number of output tiles in Height direction
    val outputHCount = Input(UInt(hCountWidth.W))
    // - number of output tiles in Width direction
    val outputWCount = Input(UInt(wCountWidth.W))
    // - (outputC / B * inputC / B * kernelY * kernelX)
    val kernelBlockCount = Input(UInt(blockCountWidth.W))

    // Avalon test interface
    val avalonMasterAddress = Output(UInt(avalonAddrWidth.W))
    val avalonMasterRead = Output(Bool())
    val avalonMasterBurstCount = Output(UInt(10.W))
    val avalonMasterWaitRequest = Input(Bool())

    val avalonMasterReadDataValid = Input(Bool())
    val avalonMasterReadData = Input(UInt(avalonDataWidth.W))

    // WMem test interface
    val wmemRead = Input(ReadPort(wAddrWidth))
    val wmemQ = Output(Vec(b, UInt(1.W)))

    // AMem sync test interface
    val wRawZero = Output(Bool())
    val wRawDec = Input(Bool())
    val wWarInc = Input(Bool())
  })

  val wmem = Module(new PackedBlockRam(b, wmemSize, 1))
  wmem.io.read := io.wmemRead
  io.wmemQ := wmem.io.q

  val maxCount = wmemSize / b
  val wSemaPair = Module(new SemaphorePair(Chisel.log2Up(maxCount) + 1, 0, maxCount))
  io.wRawZero := wSemaPair.io.consumer.rawZero
  wSemaPair.io.consumer.rawDec := io.wRawDec
  wSemaPair.io.consumer.warInc := io.wWarInc

  val wdma = Module(new WDma(b, avalonAddrWidth, avalonDataWidth, wAddrWidth))
  wdma.io.start := io.start
  wdma.io.startAddress := io.startAddress
  wdma.io.outputHCount := io.outputHCount
  wdma.io.outputWCount := io.outputWCount
  wdma.io.kernelBlockCount := io.kernelBlockCount

  wdma.io.wWarZero := wSemaPair.io.producer.warZero
  wSemaPair.io.producer.warDec := wdma.io.wWarDec
  wSemaPair.io.producer.rawInc := wdma.io.wRawInc

  wmem.io.write := wdma.io.wmemWrite

  io.avalonMasterAddress := wdma.io.avalonMasterAddress
  io.avalonMasterRead := wdma.io.avalonMasterRead
  io.avalonMasterBurstCount := wdma.io.avalonMasterBurstCount
  wdma.io.avalonMasterWaitRequest := io.avalonMasterWaitRequest
  wdma.io.avalonMasterReadDataValid := io.avalonMasterReadDataValid
  wdma.io.avalonMasterReadData := io.avalonMasterReadData
}

class WDmaTestWMemWriting(dut: WDmaTestModule, b: Int, avalonDataWidth: Int, wmemSize: Int, hCount: Int, wCount: Int, blockCount: Int, avalonAcceptLatency: Int, avalonResponseLatency: Int) extends PeekPokeTester(dut) {
  val kernelDataWidth = b
  val readsPerKenrel = kernelDataWidth / avalonDataWidth
  // input as avalon data
  val inputMemory = Seq.fill(blockCount * readsPerKenrel * b)(scala.util.Random.nextLong() & ((0x1L << avalonDataWidth) - 1))
  val avalonDataByteWidth = avalonDataWidth / 8

  object AvalonStub {
    class Request(var addr: Int, var burst: Int) {
    }

    var leftUntilAccept = avalonAcceptLatency
    var leftUntilResponse = avalonResponseLatency

    val requests = mutable.Queue[Request]()

    def next() {
      // serve pending requests
      if (requests.isEmpty) {
        poke(dut.io.avalonMasterReadDataValid, false)
      }
      else if (leftUntilResponse > 0) {
        poke(dut.io.avalonMasterReadDataValid, false)
        leftUntilResponse -= 1
      }
      else {
        val req = requests.front
        poke(dut.io.avalonMasterReadDataValid, true)
        poke(dut.io.avalonMasterReadData, inputMemory(req.addr / avalonDataByteWidth))
        req.burst -= 1
        req.addr += avalonDataByteWidth
        if (req.burst == 0)
          requests.dequeue()
        leftUntilResponse = avalonResponseLatency
      }
      // queue new requests if any and latency exceeded
      poke(dut.io.avalonMasterWaitRequest, false)
      if (peek(dut.io.avalonMasterRead).toInt == 1) {
        if (leftUntilAccept > 0) {
          poke(dut.io.avalonMasterWaitRequest, true)
          leftUntilAccept -= 1
        }
        else {
          requests.enqueue(new Request(peek(dut.io.avalonMasterAddress).toInt, peek(dut.io.avalonMasterBurstCount).toInt))
          leftUntilAccept = avalonAcceptLatency
        }
      }
    }
  }

  def doStep() = {
    AvalonStub.next()
    step(1)
  }

  def expectedData(kernel: Int) = {
    var expected = 0
    for (r <- 0 until readsPerKenrel) {
      expected |= inputMemory(kernel * readsPerKenrel + r) << (r * avalonDataWidth)
    }
    expected
  }

  poke(dut.io.start, true)
  poke(dut.io.startAddress, 0)
  poke(dut.io.outputHCount, hCount)
  poke(dut.io.outputWCount, wCount)
  poke(dut.io.kernelBlockCount, blockCount)

  poke(dut.io.wWarInc, false)
  poke(dut.io.wRawDec, false)
  var wmemAddr = 0
  for (h <- 0 until hCount) {
    for (w <- 0 until wCount) {
      for (block <- 0 until blockCount) {
        while (peek(dut.io.wRawZero) != 0) {
          doStep()
        }
        poke(dut.io.wRawDec, true)
        val blockStart = block * b
        for (kernel <- blockStart until (blockStart + b)) {
          poke(dut.io.wmemRead.addr, wmemAddr)
          poke(dut.io.wmemRead.enable, true)
          wmemAddr = (wmemAddr + 1) % wmemSize
          doStep()
          poke(dut.io.wRawDec, false)
          val expected = expectedData(kernel)
          for (k <- 0 until b) {
            expect(dut.io.wmemQ(k), (expected >> k) & 0x1)
          }
        }
        poke(dut.io.wWarInc, true)
        doStep()
        poke(dut.io.wWarInc, false)
      }
    }
  }
}

object WDmaTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val wmemSize = b * 4

    val hCount = 2
    val wCount = 2
    val blockCount = 8

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()

    var ok = true
    breakable {
      for (avalonAcceptLatency <- 0 until 1) {
        for (avalonResponseLatency <- 0 until 1) {
          for (avalonDataWidth <- List(b, b / 2)) {
            ok &= Driver.execute(driverArgs, () => new WDmaTestModule(b, avalonDataWidth, wmemSize))(
              dut => new WDmaTestWMemWriting(dut, b, avalonDataWidth, wmemSize, hCount, wCount, blockCount, avalonAcceptLatency, avalonResponseLatency))
            if (!ok) {
              break
            }
          }
        }
      }
    }
  }
}
