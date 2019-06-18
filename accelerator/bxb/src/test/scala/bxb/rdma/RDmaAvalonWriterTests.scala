package bxb.rdma

import scala.collection._
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyRequest(val addr: Int, val burst: Int) {
}

class DummyWriteRequestGenerator(b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int, startAddr: Int = 0) {
  val depthByteWidth = b * 2 / 8 // assume that bus width matches data size nicely
  val requestSeq = mutable.ArrayBuffer[DummyRequest]()
  val rowToRowDistance = inputWidth - tileWidth + 1
  for (y <- 0 until tileHeight) {
    for (x <- 0 until tileWidth by maxBurst) {
      val addr = startAddr + (y * inputWidth * depthByteWidth + x * depthByteWidth)
      val burst = if (x + maxBurst <= tileWidth) maxBurst else (tileWidth % maxBurst)
      requestSeq += new DummyRequest(addr, burst)
    }
  }
}

class RDmaAvalonWriterTestRequestSequence(dut: RDmaAvalonWriter, b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyWriteRequestGenerator(b, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileRowToRowDistance, ref.rowToRowDistance)
  poke(dut.io.avalonMasterWaitRequest, false)
  poke(dut.io.readerReady, true)
  expect(dut.io.avalonMasterWrite, false)
  for (req <- ref.requestSeq) {
    // write signal supposed to remain stable untill all data is sent
    for (i <- 0 until req.burst) {
      step(1)
      if (i == 0) {
        expect(dut.io.avalonMasterAddress, req.addr)
      }
      expect(dut.io.avalonMasterWrite, true)
      expect(dut.io.avalonMasterBurstCount, req.burst)
      expect(dut.io.tileAccepted, false)
    }
    if (req == ref.requestSeq.last) {
      step(1)
      expect(dut.io.tileAccepted, true)
    }
  }
}

class RDmaAvalonWriterTestReaderReady(dut: RDmaAvalonWriter, b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyWriteRequestGenerator(b, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileRowToRowDistance, ref.rowToRowDistance)
  poke(dut.io.avalonMasterWaitRequest, false)
  poke(dut.io.readerReady, false)
  for (_ <- 0 until 10) {
    step(1)
    expect(dut.io.avalonMasterWrite, false)
  }
  poke(dut.io.readerReady, true)
  for (req <- ref.requestSeq) {
    // write signal supposed to remain stable untill all data is sent
    for (i <- 0 until req.burst) {
      step(1)
      if (i == 0) {
        expect(dut.io.avalonMasterAddress, req.addr)
      }
      expect(dut.io.avalonMasterWrite, true)
      expect(dut.io.avalonMasterBurstCount, req.burst)
      expect(dut.io.tileAccepted, false)
    }
    if (req == ref.requestSeq.last) {
      step(1)
      expect(dut.io.tileAccepted, true)
    }
  }
}

class RDmaAvalonWriterTestWaitRequest(dut: RDmaAvalonWriter, b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyWriteRequestGenerator(b, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileRowToRowDistance, ref.rowToRowDistance)
  poke(dut.io.avalonMasterWaitRequest, true)
  poke(dut.io.readerReady, true)
  var acceptDelay = 1
  expect(dut.io.avalonMasterWrite, false)
  for (req <- ref.requestSeq) {
    // write signal supposed to remain stable untill all data is sent
    for (i <- 0 until req.burst) {
      poke(dut.io.avalonMasterWaitRequest, true)
      for (t <- 0 until acceptDelay) {
        step(1)
        if (i == 0) {
          expect(dut.io.avalonMasterAddress, req.addr)
        }
        expect(dut.io.avalonMasterWrite, true)
        expect(dut.io.avalonMasterBurstCount, req.burst)
        expect(dut.io.tileAccepted, false)
      }
      acceptDelay = (acceptDelay + 2) % 10
      poke(dut.io.avalonMasterWaitRequest, false)
      step(1)
    }
    if (req == ref.requestSeq.last) {
      expect(dut.io.tileAccepted, true)
    }
  }
}

object RDmaAvalonWriterTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val inputWidth = 20
    val inputHeight = 20
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val tileCountWidth = aAddrWidth
    val avalonAddrWidth = Chisel.log2Up(inputWidth * inputWidth * b * 2 / 8)
    val avalonDataWidth = b * 2

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    breakable {
      for (maxBurst <- List(1, 2, 4)) {
        for ((tileHeight, tileWidth) <- List((2, 2), (10, 10))) {
          println(f"running with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
          require(amemSize >= tileHeight * tileWidth)
          ok &= Driver.execute(driverArgs, () => new RDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))(
            dut => new RDmaAvalonWriterTestRequestSequence(dut, b, tileHeight, tileWidth, inputWidth, maxBurst))
          ok &= Driver.execute(driverArgs, () => new RDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))(
            dut => new RDmaAvalonWriterTestReaderReady(dut, b, tileHeight, tileWidth, inputWidth, maxBurst))
          ok &= Driver.execute(driverArgs, () => new RDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))(
            dut => new RDmaAvalonWriterTestWaitRequest(dut, b, tileHeight, tileWidth, inputWidth, maxBurst))
          if (!ok) {
            println(f"failed with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
            break
          }
        }
      }
    }
  }
}
