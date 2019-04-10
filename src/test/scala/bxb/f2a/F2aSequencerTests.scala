package bxb.f2a

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyControl(val start: Boolean, val aOffset: Int, val fOffset: Int, val qOffset: Int, val aAddr: Int, val fAddr: Int, val qAddr: Int, val decFRaw: Boolean, val incFWar: Boolean, val decAWar: Boolean, val incARaw: Boolean, val decQRaw: Boolean, val incQWar: Boolean) {
}

class DummyControlSequencer(tileHeight: Int, tileWidth: Int, repeat: Int) {
  val controlSeq = mutable.ArrayBuffer[DummyControl]()
  val aOffsetSeq = mutable.ArrayBuffer[Int]()
  val vCount = tileHeight * tileWidth
  var qOffset = 0
  var fOffset = 0
  var aOffset = 0

  for (ki <- 0 until repeat) {
    qOffset += vCount
    fOffset += vCount
    aOffset += vCount
    var qAddr = qOffset
    var fAddr = fOffset
    var aAddr = aOffset
    for (i <- 0 until vCount + 2) {
      val decQRaw = (i == 0)
      val incQWar = (i == 1)
      val decFRaw = (i == 1)
      val incFWar = (i == (vCount + 1))
      val decAWar = (i == 2)
      val incARaw = (i == (vCount + 1))
      controlSeq += new DummyControl(i == 0, aOffset, fOffset, qOffset, aAddr, fAddr, qAddr, decFRaw, incFWar, decAWar, incARaw, decQRaw, incQWar)
      if ((i >= 1) && (i <= vCount - 0)) {
        fAddr += 1
      }
      if ((i >= 1) && (i <= vCount - 0)) {
        aAddr += 1
      }
    }
  }
}

class F2aSequencerTestSequence(dut: F2aSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  poke(dut.io.hCount, tileHeight)
  poke(dut.io.wCount, tileWidth)
  poke(dut.io.aWarZero, false)
  poke(dut.io.fRawZero, false)
  poke(dut.io.qRawZero, false)

  val ref = new DummyControlSequencer(tileHeight, tileWidth, 9)
  for (ctl <- ref.controlSeq) {
    poke(dut.io.qOffset, ctl.qOffset)
    poke(dut.io.fOffset, ctl.fOffset)
    poke(dut.io.aOffset, ctl.aOffset)
    step(1)
    expect(dut.io.control.syncInc.qWar, ctl.incQWar)
    expect(dut.io.control.syncInc.fWar, ctl.incFWar)
    expect(dut.io.control.syncInc.aRaw, ctl.incARaw)
    expect(dut.io.aWarDec, ctl.decAWar)
    expect(dut.io.fRawDec, ctl.decFRaw)
    expect(dut.io.qRawDec, ctl.decQRaw)
  }
}

class F2aSequencerTestAWarZero(dut: F2aSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  poke(dut.io.hCount, tileHeight)
  poke(dut.io.wCount, tileWidth)
  poke(dut.io.aWarZero, false)
  poke(dut.io.fRawZero, false)
  poke(dut.io.qRawZero, false)
  var waitDelay = 3

  val ref = new DummyControlSequencer(tileHeight, tileWidth, 9)
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      poke(dut.io.aWarZero, true)
      for (j <- 0 until waitDelay) {
        step(1)
      }
      poke(dut.io.aWarZero, false)
    }
    poke(dut.io.qOffset, ctl.qOffset)
    poke(dut.io.fOffset, ctl.fOffset)
    poke(dut.io.aOffset, ctl.aOffset)
    step(1)
    expect(dut.io.control.syncInc.qWar, ctl.incQWar)
    expect(dut.io.control.syncInc.fWar, ctl.incFWar)
    expect(dut.io.control.syncInc.aRaw, ctl.incARaw)
    expect(dut.io.aWarDec, ctl.decAWar)
    expect(dut.io.fRawDec, ctl.decFRaw)
    expect(dut.io.qRawDec, ctl.decQRaw)
  }
}

class F2aSequencerTestFRawZero(dut: F2aSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  poke(dut.io.hCount, tileHeight)
  poke(dut.io.wCount, tileWidth)
  poke(dut.io.aWarZero, false)
  poke(dut.io.fRawZero, false)
  poke(dut.io.qRawZero, false)
  var waitDelay = 3

  val ref = new DummyControlSequencer(tileHeight, tileWidth, 9)
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      poke(dut.io.fRawZero, true)
      for (j <- 0 until waitDelay) {
        step(1)
      }
      poke(dut.io.fRawZero, false)
    }
    poke(dut.io.qOffset, ctl.qOffset)
    poke(dut.io.fOffset, ctl.fOffset)
    poke(dut.io.aOffset, ctl.aOffset)
    step(1)
    expect(dut.io.control.syncInc.qWar, ctl.incQWar)
    expect(dut.io.control.syncInc.fWar, ctl.incFWar)
    expect(dut.io.control.syncInc.aRaw, ctl.incARaw)
    expect(dut.io.aWarDec, ctl.decAWar)
    expect(dut.io.fRawDec, ctl.decFRaw)
    expect(dut.io.qRawDec, ctl.decQRaw)
  }
}

class F2aSequencerTestQRawZero(dut: F2aSequencer, tileHeight: Int, tileWidth: Int) extends PeekPokeTester(dut) {
  poke(dut.io.hCount, tileHeight)
  poke(dut.io.wCount, tileWidth)
  poke(dut.io.aWarZero, false)
  poke(dut.io.fRawZero, false)
  poke(dut.io.qRawZero, false)
  var waitDelay = 3

  val ref = new DummyControlSequencer(tileHeight, tileWidth, 9)
  for (ctl <- ref.controlSeq) {
    if (ctl.start) {
      poke(dut.io.qRawZero, true)
      for (j <- 0 until waitDelay) {
        step(1)
      }
      poke(dut.io.qRawZero, false)
    }
    poke(dut.io.qOffset, ctl.qOffset)
    poke(dut.io.fOffset, ctl.fOffset)
    poke(dut.io.aOffset, ctl.aOffset)
    step(1)
    expect(dut.io.control.syncInc.qWar, ctl.incQWar)
    expect(dut.io.control.syncInc.fWar, ctl.incFWar)
    expect(dut.io.control.syncInc.aRaw, ctl.incARaw)
    expect(dut.io.aWarDec, ctl.decAWar)
    expect(dut.io.fRawDec, ctl.decFRaw)
    expect(dut.io.qRawDec, ctl.decQRaw)
  }
}

object F2aSequencerTests {
  def main(args: Array[String]): Unit = {
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    val tileHeight = 4
    val tileWidth = 2
    var ok = true
    ok &= Driver.execute(driverArgs, () => new F2aSequencer(3,10,10,10,10,10,10))(
      dut => new F2aSequencerTestSequence(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new F2aSequencer(3,10,10,10,10,10,10))(
      dut => new F2aSequencerTestAWarZero(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new F2aSequencer(3,10,10,10,10,10,10))(
      dut => new F2aSequencerTestFRawZero(dut, tileHeight, tileWidth))
    ok &= Driver.execute(driverArgs, () => new F2aSequencer(3,10,10,10,10,10,10))(
      dut => new F2aSequencerTestQRawZero(dut, tileHeight, tileWidth))
    if (!ok && args.contains("noexit"))
      System.exit(1)
  }
}
