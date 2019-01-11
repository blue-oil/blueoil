package bxb.a2f

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyControl(val aAddr: Int, val fAddr: Int, val accumulate: Boolean, val evenOdd: Int) {
}

class DummyControlSequencer(tileHeight: Int, tileWidth: Int) {
  val controlSeq = mutable.ArrayBuffer[DummyControl]()
  val aOffsetSeq = mutable.ArrayBuffer[Int]()
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
      aOffsetSeq += aOffset
      aOffset += (if (kj == 2) hCount else 1)
      for (i <- 0 until vCount) {
        for (j <- 0 until hCount) {
          controlSeq += new DummyControl(aAddr, fAddr, acc, evenOdd)
          aAddr += (if (j == hCount - 1) gap else step)
          fAddr += 1
        }
      }
      evenOdd = evenOdd ^ 0x1
    }
  }
}

class A2fSequencerTestSequence(dut: A2fSequencer) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(6, 6)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.waitReq, false)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  while (peek(dut.io.controlValid) == 0) {
    step(1)
  }
  for (i <- 0 until 3) {
    for (ctl <- ref.controlSeq) {
      expect(dut.io.control.aAddr, ctl.aAddr)
      expect(dut.io.control.fAddr, ctl.fAddr)
      expect(dut.io.control.writeEnable, true)
      expect(dut.io.control.accumulate, ctl.accumulate)
      step(1)
    }
  }
}

class A2fSequencerTestEvenOdd(dut: A2fSequencer) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(6, 6)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.waitReq, false)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  while (peek(dut.io.controlValid) == 0) {
    step(1)
  }
  for (ctl <- ref.controlSeq) {
    expect(dut.io.control.evenOdd, ctl.evenOdd)
    step(1)
  }
}

class A2fSequencerTestWaitReq(dut: A2fSequencer) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(6, 6)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.waitReq, true)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, true)
  var waitDelay = 1
  for (i <- 0 until 3) {
    for (j <- 0 until waitDelay) {
      step(1)
    }
    poke(dut.io.waitReq, false)
    waitDelay += 2
    while (peek(dut.io.controlValid) == 0) {
      step(1)
    }
    for (ctl <- ref.controlSeq) {
      expect(dut.io.control.aAddr, ctl.aAddr)
      expect(dut.io.control.fAddr, ctl.fAddr)
      expect(dut.io.control.writeEnable, true)
      expect(dut.io.control.accumulate, ctl.accumulate)
      step(1)
    }
    poke(dut.io.waitReq, true)
  }
}

class A2fSequencerTestOffsetValid(dut: A2fSequencer) extends PeekPokeTester(dut) {
  val ref = new DummyControlSequencer(6, 6)
  poke(dut.io.tileHCount, ref.hCount)
  poke(dut.io.tileVCount, ref.vCount)
  poke(dut.io.tileStep, ref.step)
  poke(dut.io.tileGap, ref.gap)
  poke(dut.io.kernelHCount, 3)
  poke(dut.io.kernelVCount, 3)
  poke(dut.io.waitReq, false)
  poke(dut.io.tileOffset, 0)
  poke(dut.io.tileOffsetValid, false)
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
      step(1)
    }
  }
}

object A2fSequencerTests {
  def main(args: Array[String]): Unit = {
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestSequence(dut))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestEvenOdd(dut))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestWaitReq(dut))
    ok &= Driver.execute(driverArgs, () => new A2fSequencer(10))(dut => new A2fSequencerTestOffsetValid(dut))
    if (!ok && args.contains("noexit"))
      System.exit(1)
  }
}
