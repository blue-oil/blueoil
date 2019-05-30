package bxb.wqdma

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class WQDmaAvalonRequesterTestSequence(dut: WQDmaAvalonRequester, b: Int, hCount: Int, wCount: Int, blockCount: Int) extends PeekPokeTester(dut) {
  poke(dut.io.start, true)
  poke(dut.io.startAddress, 0)
  poke(dut.io.outputHCount, hCount)
  poke(dut.io.outputWCount, wCount)
  poke(dut.io.blockCount, blockCount)
  poke(dut.io.writerDone, true)
  poke(dut.io.warZero, false)
  poke(dut.io.avalonMasterWaitRequest, false)
  for (h <- 0 until hCount) {
    for (w <- 0 until wCount) {
      for (block <- 0 until blockCount) {
        while (peek(dut.io.avalonMasterRead) == 0) {
          step(1)
        }
        expect(dut.io.avalonMasterRead, true)
        expect(dut.io.avalonMasterAddress, block * b * b / 8)
        expect(dut.io.avalonMasterBurstCount, b)
        step(1)
      }
    }
  }
}

class WQDmaAvalonRequesterTestWriterDone(dut: WQDmaAvalonRequester, b: Int, hCount: Int, wCount: Int, blockCount: Int) extends PeekPokeTester(dut) {
  poke(dut.io.start, true)
  poke(dut.io.startAddress, 0)
  poke(dut.io.outputHCount, hCount)
  poke(dut.io.outputWCount, wCount)
  poke(dut.io.blockCount, blockCount)
  poke(dut.io.writerDone, false)
  poke(dut.io.warZero, false)
  poke(dut.io.avalonMasterWaitRequest, false)
  var writerDelay = 1
  for (h <- 0 until hCount) {
    for (w <- 0 until wCount) {
      for (block <- 0 until blockCount) {
        poke(dut.io.writerDone, false)
        while (peek(dut.io.avalonMasterRead) == 0) {
          step(1)
        }
        expect(dut.io.avalonMasterRead, true)
        expect(dut.io.avalonMasterAddress, block * b * b / 8)
        expect(dut.io.avalonMasterBurstCount, b)
        step(1)
        for (_ <- 0 until writerDelay) {
          step(1)
        }
        writerDelay = (writerDelay + 1) % b
        poke(dut.io.writerDone, true)
        step(1)
      }
    }
  }
}

class WQDmaAvalonRequesterTestWarZero(dut: WQDmaAvalonRequester, b: Int, hCount: Int, wCount: Int, blockCount: Int) extends PeekPokeTester(dut) {
  poke(dut.io.start, true)
  poke(dut.io.startAddress, 0)
  poke(dut.io.outputHCount, hCount)
  poke(dut.io.outputWCount, wCount)
  poke(dut.io.blockCount, blockCount)
  poke(dut.io.writerDone, true)
  poke(dut.io.warZero, true)
  poke(dut.io.avalonMasterWaitRequest, false)
  var warDelay = 1
  for (h <- 0 until hCount) {
    for (w <- 0 until wCount) {
      for (block <- 0 until blockCount) {
        poke(dut.io.warZero, true)
        for (_ <- 0 until warDelay) {
          step(1)
        }
        warDelay = (warDelay + 1) % b
        poke(dut.io.warZero, false)
        step(1)
        while (peek(dut.io.avalonMasterRead) == 0) {
          step(1)
        }
        expect(dut.io.avalonMasterRead, true)
        expect(dut.io.avalonMasterAddress, block * b * b / 8)
        expect(dut.io.avalonMasterBurstCount, b)
        step(1)
      }
    }
  }
}

object WQDmaAvalonRequesterTests {
  def main(args: Array[String]): Unit = {
    val b = 32
    val avalonAddrWidth = 32
    val avalonDataWidth = b

    val hCount = 5
    val wCount = 5
    val blockCount = 16

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()

    // wdma parameters:
    val elementWidth = b * 1
    val elementsPerBlock = b

    var ok = true
    ok &= Driver.execute(driverArgs, () => new WQDmaAvalonRequester(avalonAddrWidth, avalonDataWidth, elementWidth, elementsPerBlock))(
      dut => new WQDmaAvalonRequesterTestSequence(dut, b, hCount, wCount, blockCount))
    ok &= Driver.execute(driverArgs, () => new WQDmaAvalonRequester(avalonAddrWidth, avalonDataWidth, elementWidth, elementsPerBlock))(
      dut => new WQDmaAvalonRequesterTestWriterDone(dut, b, hCount, wCount, blockCount))
    ok &= Driver.execute(driverArgs, () => new WQDmaAvalonRequester(avalonAddrWidth, avalonDataWidth, elementWidth, elementsPerBlock))(
      dut => new WQDmaAvalonRequesterTestWarZero(dut, b, hCount, wCount, blockCount))
  }
}
