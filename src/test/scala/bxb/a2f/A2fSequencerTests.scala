package bxb.a2f

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyControl(val aAddr: Int, val fAddr: Int, val accumulate: Boolean, val evenOdd: Int, val decMRaw: Boolean, val incMWar: Boolean, val decARaw: Boolean, val incAWar: Boolean, val incFRaw: Boolean) {
}

class DummyControlSequencer(tileHeight: Int, tileWidth: Int) {
  val controlSeq = mutable.ArrayBuffer[DummyControl]()
  val vCount = tileHeight - 2
  val hCount = tileWidth - 2
  val step = 1
  val gap = 3
  private var aOffset = 0
  private var evenOdd = 0
  for (ki <- 0 until 3) {
    for (kj <- 0 until 3) {
      val acc = !(ki == 0 && kj == 0)
      var aAddr = aOffset
      var fAddr = 0
      aOffset += (if (kj == 2) hCount else 1)
      for (i <- 0 until vCount) {
        for (j <- 0 until hCount) {
          val decARaw = (ki == 0 && kj == 0 && i == 0 && j == 0)
          val incAWar = (ki == 2 && kj == 2 && i == vCount - 1 && j == hCount - 1)
          val decMRaw = (i == 0 && j == 0) // decrement semaphore before starting 1x1 convolution
          val incMWar = (i == vCount - 1 && j == hCount - 1) // increment semaphore when 1x1 done
          val incFRaw = (ki == 2 && kj == 2 && i == vCount - 1 && j == hCount - 1)
          controlSeq += new DummyControl(aAddr, fAddr, acc, evenOdd, decMRaw, incMWar, decARaw, incAWar, incFRaw)
          aAddr += (if (j == hCount - 1) gap else step)
          fAddr += 1
        }
      }
      evenOdd = evenOdd ^ 0x1
    }
  }
}

class A2fSequencerTestSequence(dut: A2fSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(tileHeight, tileWidth)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, false)
  while (peek(dut.io.controlValid) == 0) {
    step(1)
  }
  for (i <- 0 until 3) {
    for (ctl <- ref.controlSeq) {
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
}

class A2fSequencerTestEvenOdd(dut: A2fSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(tileHeight, tileWidth)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, false)
  while (peek(dut.io.controlValid) == 0) {
    step(1)
  }
  for (ctl <- ref.controlSeq) {
    expect(dut.io.control.evenOdd, ctl.evenOdd)
    step(1)
  }
}

class A2fSequencerTestARawZero(dut: A2fSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(tileHeight, tileWidth)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, true)
  var waitDelay = 1
  for (i <- 0 until 3) {
    for (j <- 0 until waitDelay) {
      step(1)
    }
    poke(dut.io.aRawZero, false)
    waitDelay += 2
    while (peek(dut.io.controlValid) == 0) {
      step(1)
    }
    for (ctl <- ref.controlSeq) {
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
    poke(dut.io.aRawZero, true)
  }
}

class A2fSequencerTestMRawZero(dut: A2fSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(tileHeight, tileWidth)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  poke(dut.io.mRawZero, true)
  poke(dut.io.aRawZero, false)
  var waitDelay = 1
  for (i <- 0 until 3) {
    for (j <- 0 until waitDelay) {
      step(1)
    }
    poke(dut.io.mRawZero, false)
    waitDelay += 2
    while (peek(dut.io.controlValid) == 0) {
      step(1)
    }
    for (ctl <- ref.controlSeq) {
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
    poke(dut.io.mRawZero, true)
  }
}

class A2fSequencerTestOffsetValid(dut: A2fSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(tileHeight, tileWidth)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, false)
  poke(dut.io.mRawZero, false)
  poke(dut.io.aRawZero, false)
  var waitDelay = 1
  for (i <- 0 until 3) {
    for (j <- 0 until waitDelay) {
      step(1)
    }
    poke(dut.io.tileOffsetValid, true)
    waitDelay += 2
    while (peek(dut.io.controlValid) == 0) {
      expect(dut.io.control.writeEnable, false)
      step(1)
    }
    poke(dut.io.tileOffsetValid, false)
    for (ctl <- ref.controlSeq) {
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
}

object A2fSequencerTests {
  def main(args: Array[String]): Unit = {

    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()

    val tileHeight = 6
    val tileWidth = 6

    var ok = true
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestSequence(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestEvenOdd(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestARawZero(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestMRawZero(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestOffsetValid(dut, tileHeight, tileWidth))
    if (!ok && args.contains("noexit"))
      System.exit(1)
  }
}
