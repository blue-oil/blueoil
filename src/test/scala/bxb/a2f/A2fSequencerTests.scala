package bxb.a2f 

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyControl(val start: Boolean, val aAddr: Int, val fAddr: Int, val accumulate: Boolean, val evenOdd: Int, val decMRaw: Boolean, val incMWar: Boolean, val decARaw: Boolean, val incAWar: Boolean, val incFRaw: Boolean) {
}

class DummyControlSequencer(repeats: Int, b: Int, amemSize: Int, tileHeight: Int, tileWidth: Int, inputChannels: Int) {
  val controlSeq = mutable.ArrayBuffer[DummyControl]()
  val vCount = tileHeight - 2
  val hCount = tileWidth - 2
  val cCount = inputChannels / b
  val step = 1
  val gap = 3
  private val amemHalf = amemSize / 2
  private var aOffset = 0
  private var fOffset = 0
  private var evenOdd = 0
  for (r <- 0 until repeats) {
    for (c <- 0 until cCount) {
      for (ki <- 0 until 3) {
        for (kj <- 0 until 3) {
          val acc = !(ki == 0 && kj == 0)
          var aAddr = aOffset
          var fAddr = fOffset
          aOffset += (if (kj == 2) hCount else 1)
          for (i <- 0 until vCount) {
            for (j <- 0 until hCount) {
              val start = (c == 0 && ki == 0 && kj == 0 && i == 0 && j == 0)
              val decARaw = (ki == 0 && kj == 0 && i == 0 && j == 0)
              val incAWar = (ki == 2 && kj == 2 && i == vCount - 1 && j == hCount - 1)
              val decMRaw = (i == 0 && j == 0) // decrement semaphore before starting 1x1 convolution
              val incMWar = (i == vCount - 1 && j == hCount - 1) // increment semaphore when 1x1 done
              val incFRaw = (ki == 2 && kj == 2 && i == vCount - 1 && j == hCount - 1)
              controlSeq += new DummyControl(start, aAddr, fAddr, acc, evenOdd, decMRaw, incMWar, decARaw, incAWar, incFRaw)
              aAddr += (if (j == hCount - 1) gap else step)
              fAddr += 1
            }
          }
          evenOdd = evenOdd ^ 0x1
        }
      }
      aOffset = if (aOffset < amemHalf) amemHalf else 0 // use another half
      fOffset = if (fOffset < amemHalf) amemHalf else 0 // use another half
    }
  }
}

class A2fSequencerTestSequence(dut: A2fSequencer, b: Int, amemSize: Int, tileHeight: Int, tileWidth: Int, inputChannels: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(3, b, amemSize, tileHeight, tileWidth, inputChannels)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileValid, true)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, false)
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      while (peek(dut.io.controlValid) == 0) {
        step(1)
      }
    }
    expect(dut.io.control.aAddr, ctl.aAddr)
    expect(dut.io.control.fAddr, ctl.fAddr)
    expect(dut.io.control.writeEnable, true)
    expect(dut.io.control.accumulate, ctl.accumulate)
    expect(dut.io.control.syncInc.aWar, ctl.incAWar)
    expect(dut.io.control.syncInc.mWar, ctl.incMWar)
    expect(dut.io.control.syncInc.fRaw, ctl.incFRaw)
    expect(dut.io.aRawDec, ctl.decARaw)
    expect(dut.io.mRawDec, ctl.decMRaw)
    step(1)
  }
}

class A2fSequencerTestEvenOdd(dut: A2fSequencer, b: Int, amemSize: Int, tileHeight: Int, tileWidth: Int, inputChannels: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(3, b, amemSize, tileHeight, tileWidth, inputChannels)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileValid, true)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, false)
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      while (peek(dut.io.controlValid) == 0) {
        step(1)
      }
    }
    expect(dut.io.control.evenOdd, ctl.evenOdd)
    step(1)
  }
}

class A2fSequencerTestARawZero(dut: A2fSequencer, b: Int, amemSize: Int, tileHeight: Int, tileWidth: Int, inputChannels: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(3, b, amemSize, tileHeight, tileWidth, inputChannels)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileValid, true)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, true)
  var waitDelay = 1
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      poke(dut.io.aRawZero, true)
      for (j <- 0 until waitDelay) {
        step(1)
      }
      poke(dut.io.aRawZero, false)
      waitDelay += 2
      while (peek(dut.io.controlValid) == 0) {
        step(1)
      }
    }
    expect(dut.io.control.aAddr, ctl.aAddr)
    expect(dut.io.control.fAddr, ctl.fAddr)
    expect(dut.io.control.writeEnable, true)
    expect(dut.io.control.accumulate, ctl.accumulate)
    expect(dut.io.control.syncInc.aWar, ctl.incAWar)
    expect(dut.io.control.syncInc.mWar, ctl.incMWar)
    expect(dut.io.control.syncInc.fRaw, ctl.incFRaw)
    expect(dut.io.aRawDec, ctl.decARaw)
    expect(dut.io.mRawDec, ctl.decMRaw)
    step(1)
  }
}

class A2fSequencerTestMRawZero(dut: A2fSequencer, b: Int, amemSize: Int, tileHeight: Int, tileWidth: Int, inputChannels: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(3, b, amemSize, tileHeight, tileWidth, inputChannels)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileValid, true)
  poke(dut.io.mRawZero, true)
  poke(dut.io.aRawZero, false)
  var waitDelay = 1
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      poke(dut.io.mRawZero, true)
      for (j <- 0 until waitDelay) {
        step(1)
      }
      poke(dut.io.mRawZero, false)
      waitDelay += 2
      while (peek(dut.io.controlValid) == 0) {
        step(1)
      }
    }
    expect(dut.io.control.aAddr, ctl.aAddr)
    expect(dut.io.control.fAddr, ctl.fAddr)
    expect(dut.io.control.writeEnable, true)
    expect(dut.io.control.accumulate, ctl.accumulate)
    expect(dut.io.control.syncInc.aWar, ctl.incAWar)
    expect(dut.io.control.syncInc.mWar, ctl.incMWar)
    expect(dut.io.control.syncInc.fRaw, ctl.incFRaw)
    expect(dut.io.aRawDec, ctl.decARaw)
    expect(dut.io.mRawDec, ctl.decMRaw)
    step(1)
  }
}

class A2fSequencerTestOffsetValid(dut: A2fSequencer, b: Int, amemSize: Int, tileHeight: Int, tileWidth: Int, inputChannels: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(3, b, amemSize, tileHeight, tileWidth, inputChannels)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileValid, false)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, false)
  var waitDelay = 1
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      for (j <- 0 until waitDelay) {
        step(1)
      }
      poke(dut.io.tileValid, true)
      waitDelay += 2
      while (peek(dut.io.controlValid) == 0) {
        expect(dut.io.control.writeEnable, false)
        step(1)
      }
      poke(dut.io.tileValid, false)
    }
    expect(dut.io.control.aAddr, ctl.aAddr)
    expect(dut.io.control.fAddr, ctl.fAddr)
    expect(dut.io.control.writeEnable, true)
    expect(dut.io.control.accumulate, ctl.accumulate)
    expect(dut.io.aRawDec, ctl.decARaw)
    expect(dut.io.mRawDec, ctl.decMRaw)
    expect(dut.io.control.syncInc.aWar, ctl.incAWar)
    expect(dut.io.control.syncInc.mWar, ctl.incMWar)
    expect(dut.io.control.syncInc.fRaw, ctl.incFRaw)
    step(1)
  }
}

object A2fSequencerTests {
  def main(args: Array[String]): Unit = {

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()

    val b = 4
    val amemSize = 1024

    val tileHeight = 6
    val tileWidth = 6
    val inputChannels = b * 1

    var ok = true
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestSequence(dut, b, amemSize, tileHeight, tileWidth, inputChannels))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestEvenOdd(dut, b, amemSize, tileHeight, tileWidth, inputChannels))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestARawZero(dut, b, amemSize, tileHeight, tileWidth, inputChannels))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestMRawZero(dut, b, amemSize, tileHeight, tileWidth, inputChannels))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestOffsetValid(dut, b, amemSize, tileHeight, tileWidth, inputChannels))
    if (!ok && args.contains("noexit"))
      System.exit(1)
  }
}
