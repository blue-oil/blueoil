package bxb.wqdma

import scala.collection._
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

import bxb.avalon.{ReadMasterIO}
import bxb.memory.{PackedBlockRam, ReadPort}
import bxb.sync.{SemaphorePair}

class QDmaTestModule(b: Int, qmemSize: Int) extends Module {
  val avalonAddrWidth = 32
  val avalonDataWidth = 64
  val wAddrWidth = Chisel.log2Up(qmemSize)
  require(Chisel.isPow2(qmemSize))
  require(qmemSize % b == 0)
  val io = IO(new Bundle {
    val start = Input(Bool())

    val parameters = Input(WQDmaParameters(avalonAddrWidth))

    // Avalon test interface
    val avalonMaster = ReadMasterIO(avalonAddrWidth, avalonDataWidth)

    // QMem test interface
    val qmemRead = Input(ReadPort(wAddrWidth))
    val qmemQ = Output(Vec(b, UInt(40.W)))

    // QMem sync test interface
    val qRawZero = Output(Bool())
    val qRawDec = Input(Bool())
    val qWarInc = Input(Bool())
  })

  val qmem = Module(new PackedBlockRam(b, qmemSize, 40))
  qmem.io.read := io.qmemRead
  io.qmemQ := qmem.io.q

  val maxCount = qmemSize / b
  val qSemaPair = Module(new SemaphorePair(Chisel.log2Up(maxCount) + 1, 0, maxCount))
  io.qRawZero := qSemaPair.io.consumer.rawZero
  qSemaPair.io.consumer.rawDec := io.qRawDec
  qSemaPair.io.consumer.warInc := io.qWarInc

  val qdma = Module(new QDma(b, avalonAddrWidth, avalonDataWidth, wAddrWidth))
  qdma.io.start := io.start
  qdma.io.parameters := io.parameters

  qSemaPair.io.producer <> qdma.io.qSync

  qmem.io.write := qdma.io.qmemWrite

  io.avalonMaster <> qdma.io.avalonMaster
}

class QDmaTestQMemWriting(dut: QDmaTestModule, b: Int, qmemSize: Int, hCount: Int, wCount: Int, blockCount: Int, avalonAcceptLatency: Int, avalonResponseLatency: Int) extends PeekPokeTester(dut) {
  val avalonDataWidth = 64
  val avalonDataByteWidth = avalonDataWidth / 8
  val thresholdPackWidthRaw = b * 4 * 16
  val readsPerPack = thresholdPackWidthRaw / avalonDataWidth
  // input as avalon data
  val inputMemory = Seq.fill(blockCount * readsPerPack)(scala.util.Random.nextLong())

  object AvalonStub {
    class Request(var addr: Int, var burst: Int) {
    }

    var leftUntilAccept = avalonAcceptLatency
    var leftUntilResponse = avalonResponseLatency

    val requests = mutable.Queue[Request]()

    def next() {
      // serve pending requests
      if (requests.isEmpty) {
        poke(dut.io.avalonMaster.readDataValid, false)
      }
      else if (leftUntilResponse > 0) {
        poke(dut.io.avalonMaster.readDataValid, false)
        leftUntilResponse -= 1
      }
      else {
        val req = requests.front
        poke(dut.io.avalonMaster.readDataValid, true)
        poke(dut.io.avalonMaster.readData, inputMemory(req.addr / avalonDataByteWidth))
        req.burst -= 1
        req.addr += avalonDataByteWidth
        if (req.burst == 0)
          requests.dequeue()
        leftUntilResponse = avalonResponseLatency
      }
      // queue new requests if any and latency exceeded
      poke(dut.io.avalonMaster.waitRequest, false)
      if (peek(dut.io.avalonMaster.read).toInt == 1) {
        if (leftUntilAccept > 0) {
          poke(dut.io.avalonMaster.waitRequest, true)
          leftUntilAccept -= 1
        }
        else {
          requests.enqueue(new Request(peek(dut.io.avalonMaster.address).toInt, peek(dut.io.avalonMaster.burstCount).toInt))
          leftUntilAccept = avalonAcceptLatency
        }
      }
    }
  }

  def doStep() = {
    AvalonStub.next()
    step(1)
  }

  def expectedData(start: Int) = {
    var expected = mutable.ArrayBuffer[Long]()
    val chunks = (0 until readsPerPack).map(w => {
      val word = inputMemory(start * readsPerPack + w)
      (0 until 4).map(t => (word >> (t * 16)) & 0xFFFF)
    }).toSeq
    val mask = (0x1L << 13) - 1
    for (chunk <- chunks) {
      expected += ((((chunk(3) >> 15) & 0x1) << 13 * 3)
                | ((chunk(2) & mask) << 13 * 2)
                | ((chunk(1) & mask) << 13 * 1)
                | (chunk(0) & mask))
    }
    expected
  }

  poke(dut.io.start, true)
  poke(dut.io.parameters.startAddress, 0)
  poke(dut.io.parameters.outputHCount, hCount)
  poke(dut.io.parameters.outputWCount, wCount)
  poke(dut.io.parameters.blockCount, blockCount)

  poke(dut.io.qWarInc, false)
  poke(dut.io.qRawDec, false)
  var qmemAddr = 0
  for (h <- 0 until hCount) {
    for (w <- 0 until wCount) {
      for (block <- 0 until blockCount) {
        while (peek(dut.io.qRawZero) != 0) {
          doStep()
        }
        poke(dut.io.qRawDec, true)
        poke(dut.io.qmemRead.addr, qmemAddr)
        poke(dut.io.qmemRead.enable, true)
        qmemAddr = (qmemAddr + 1) % qmemSize
        doStep()
        poke(dut.io.qRawDec, false)
        val expected = expectedData(block)
        for ((data, i) <- expected.zipWithIndex) {
          expect(dut.io.qmemQ(i), data)
        }
        poke(dut.io.qWarInc, true)
        doStep()
        poke(dut.io.qWarInc, false)
      }
    }
  }
}

object QDmaTests {
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
          ok &= Driver.execute(driverArgs, () => new QDmaTestModule(b, wmemSize))(
            dut => new QDmaTestQMemWriting(dut, b, wmemSize, hCount, wCount, blockCount, avalonAcceptLatency, avalonResponseLatency))
          if (!ok) {
            break
          }
        }
      }
    }
  }
}
