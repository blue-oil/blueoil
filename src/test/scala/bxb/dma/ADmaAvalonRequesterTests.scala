package bxb.dma

import scala.collection._
import util.control.Breaks._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyRequest(var addr: Int, var burst: Int) {
}

object DummyRequestGenerator {
  def computeRowToRowDistance(inputWidth: Int, tileWidth: Int, maxBurst: Int) = {
    if (tileWidth % maxBurst == 0) {
      (inputWidth - tileWidth + maxBurst)
    }
    else {
      (inputWidth - tileWidth + tileWidth % maxBurst)
    }
  }
}

// generates request sequence for one tile assuming tile located in the
// DDR starting from address 0 using blocked layout
class DummyRequestGenerator(b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int, startAddr: Int = 0) {
  val depthByteWidth = b * 2 / 8 // assume that bus width matches data size nicely
  val requestSeq = mutable.ArrayBuffer[DummyRequest]()
  val rowToRowDistance = DummyRequestGenerator.computeRowToRowDistance(inputWidth, tileWidth, maxBurst)
  for (y <- 0 until tileHeight) {
    for (x <- 0 until tileWidth by maxBurst) {
      val addr = startAddr + (y * inputWidth * depthByteWidth + x * depthByteWidth)
      val burst = if (x + maxBurst <= tileWidth) maxBurst else (tileWidth % maxBurst)
      requestSeq += new DummyRequest(addr, burst)
    }
  }
}

class ADmaAvalonRequesterTestRequestSequence(dut: ADmaAvalonRequester, b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyRequestGenerator(b, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileRowToRowDistance, ref.rowToRowDistance)
  poke(dut.io.avalonMasterWaitRequest, false)
  expect(dut.io.avalonMasterRead, false)
  for (req <- ref.requestSeq) {
    step(1)
    expect(dut.io.avalonMasterAddress, req.addr)
    expect(dut.io.avalonMasterRead, true)
    expect(dut.io.avalonMasterBurstCount, req.burst)
    expect(dut.io.tileAccepted, req == ref.requestSeq.last)
  }
}

class ADmaAvalonRequesterTestWaitRequest(dut: ADmaAvalonRequester, b: Int, tileHeight: Int, tileWidth: Int, inputWidth: Int, maxBurst: Int) extends PeekPokeTester(dut) {
  val ref = new DummyRequestGenerator(b, tileHeight, tileWidth, inputWidth, maxBurst)
  poke(dut.io.tileValid, true)
  poke(dut.io.tileStartAddress, 0)
  poke(dut.io.tileHeight, tileHeight)
  poke(dut.io.tileWidth, tileWidth)
  poke(dut.io.tileRowToRowDistance, ref.rowToRowDistance)
  var acceptDelay = 1
  expect(dut.io.avalonMasterRead, false)
  for (req <- ref.requestSeq) {
    poke(dut.io.avalonMasterWaitRequest, true)
    for (t <- 0 until acceptDelay) {
      step(1)
      expect(dut.io.avalonMasterAddress, req.addr)
      expect(dut.io.avalonMasterRead, true)
      expect(dut.io.avalonMasterBurstCount, req.burst)
    }
    acceptDelay = (acceptDelay + 2) % 64
    poke(dut.io.avalonMasterWaitRequest, false)
    expect(dut.io.tileAccepted, req == ref.requestSeq.last)
    step(1)
  }
}

object ADmaAvalonRequesterTests {
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
          ok &= Driver.execute(driverArgs, () => new ADmaAvalonRequester(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))(
            dut => new ADmaAvalonRequesterTestRequestSequence(dut, b, tileHeight, tileWidth, inputWidth, maxBurst))
          ok &= Driver.execute(driverArgs, () => new ADmaAvalonRequester(avalonAddrWidth, avalonDataWidth, tileCountWidth, maxBurst))(
            dut => new ADmaAvalonRequesterTestWaitRequest(dut, b, tileHeight, tileWidth, inputWidth, maxBurst))
          if (!ok)
            break
        }
      }
    }
  }
}
