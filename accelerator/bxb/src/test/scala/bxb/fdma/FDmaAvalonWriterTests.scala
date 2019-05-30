package bxb.fdma

import scala.collection._
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyRequest(val addr: Int, val burst: Int) {
}

class DummyWriteRequestGenerator(avalonDataWidth: Int, depthDataWidth: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int, startAddr: Int = 0) {
  val depthByteWidth = depthDataWidth / 8
  val wordsPerElement = depthDataWidth / avalonDataWidth
  val requestSeq = mutable.ArrayBuffer[DummyRequest]()
  val rowToRowDistance = (inputWidth - tileWidth) * wordsPerElement + 1
  for (y <- 0 until tileHeight) {
    for (x <- 0 until tileWidth by maxBurst / wordsPerElement) {
      val addr = startAddr + (y * inputWidth * depthByteWidth + x * depthByteWidth)
      val burst = if (x + maxBurst / wordsPerElement <= tileWidth) maxBurst else (tileWidth * wordsPerElement % maxBurst)
      requestSeq += new DummyRequest(addr, burst)
    }
  }
}

class FDmaAvalonWriterTestRequestSequence(dut: FDmaAvalonWriter, avalonDataWidth: Int, depthDataWidth: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyWriteRequestGenerator(avalonDataWidth, depthDataWidth, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileWordRowToRowDistance, ref.rowToRowDistance)
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

class FDmaAvalonWriterTestReaderReady(dut: FDmaAvalonWriter, avalonDataWidth: Int, depthDataWidth: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyWriteRequestGenerator(avalonDataWidth, depthDataWidth, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileWordRowToRowDistance, ref.rowToRowDistance)
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

class FDmaAvalonWriterTestWaitRequest(dut: FDmaAvalonWriter, avalonDataWidth: Int, depthDataWidth: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyWriteRequestGenerator(avalonDataWidth, depthDataWidth, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileWordRowToRowDistance, ref.rowToRowDistance)
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

object FDmaAvalonWriterTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val inputWidth = 20
    val inputHeight = 20
    val amemSize = 32 * 32
    val aAddrWidth = Chisel.log2Up(amemSize)
    val tileCountWidth = aAddrWidth
    val avalonAddrWidth = Chisel.log2Up(inputWidth * inputWidth * b * 16 / 8)

    val depthDataWidth = b * 16
    val avalonDataWidth = depthDataWidth / 4

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true

    breakable {
      for (avalonDataWidth <- List(depthDataWidth / 2, depthDataWidth / 4)) {
        for (maxBurst <- List(4, 8)) {
          for ((tileHeight, tileWidth) <- List((2, 2), (10, 10))) {
            println(f"running with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
            require(amemSize >= tileHeight * tileWidth)
            ok &= Driver.execute(driverArgs, () => new FDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, depthDataWidth, tileCountWidth, maxBurst))(
              dut => new FDmaAvalonWriterTestRequestSequence(dut, avalonDataWidth, depthDataWidth, tileHeight, tileWidth, inputWidth, maxBurst))
            ok &= Driver.execute(driverArgs, () => new FDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, depthDataWidth, tileCountWidth, maxBurst))(
              dut => new FDmaAvalonWriterTestReaderReady(dut, avalonDataWidth, depthDataWidth, tileHeight, tileWidth, inputWidth, maxBurst))
            ok &= Driver.execute(driverArgs, () => new FDmaAvalonWriter(avalonAddrWidth, avalonDataWidth, depthDataWidth, tileCountWidth, maxBurst))(
              dut => new FDmaAvalonWriterTestWaitRequest(dut, avalonDataWidth, depthDataWidth, tileHeight, tileWidth, inputWidth, maxBurst))
            if (!ok) {
              println(f"failed with maxBurst:${maxBurst} tileHeight:${tileHeight} tileWidth:${tileWidth}")
              break
            }
          }
        }
      }
    }
  }
}
