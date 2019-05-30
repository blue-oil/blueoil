package bxb.w2m

import scala.collection._

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}

class DummyW2mControl(val chunkStart: Boolean, val wAddr: Int, val mWeEven: Boolean, val mWeOdd: Boolean, val decMWar: Boolean, val incMRaw: Boolean, val decWRaw: Boolean, val incWWar: Boolean) {
}

class DummyW2mControlSequencer(val b: Int, val wMemSize: Int) {
  val controlSeq = mutable.ArrayBuffer[DummyW2mControl]()
  private var wAddr = 0
  private var mWeEven = true
  private var mWeOdd = false
  for (i <- 0 until (3 * wMemSize)) {
    val chunkStart = (i % b == 0)
    val decMWar = chunkStart
    val decWRaw = chunkStart
    val chunkEnd = (i % b == b - 1)
    val incMRaw = chunkEnd
    val incWWar = chunkEnd
    controlSeq += new DummyW2mControl(chunkStart, wAddr, mWeEven, mWeOdd, decMWar, incMRaw, decWRaw, incWWar)
    if ((i + 1) % b == 0) {
      mWeEven = !mWeEven
      mWeOdd = !mWeOdd
    }
    wAddr += 1
    // should overflow
    if (wAddr == wMemSize) {
      wAddr = 0
    }
  }
}

class W2mSequencerTestSequence(dut: W2mSequencer, val b: Int, val wMemSize: Int) extends PeekPokeTester(dut) {
  val ref = new DummyW2mControlSequencer(b, wMemSize)
  poke(dut.io.mWarZero, false)
  poke(dut.io.wRawZero, false)
  for (ctl <- ref.controlSeq) {
    expect(dut.io.control.wAddr, ctl.wAddr)
    expect(dut.io.control.mWe(0), ctl.mWeEven)
    expect(dut.io.control.mWe(1), ctl.mWeOdd)
    expect(dut.io.control.mRawInc, ctl.incMRaw)
    expect(dut.io.control.wWarInc, ctl.incWWar)
    expect(dut.io.mWarDec, ctl.decMWar)
    expect(dut.io.wRawDec, ctl.decWRaw)
    step(1)
  }
}

class W2mSequencerTestMWarZero(dut: W2mSequencer, val b: Int, val wMemSize: Int) extends PeekPokeTester(dut) {
  val ref = new DummyW2mControlSequencer(b, wMemSize)
  var waitDelay = 1
  poke(dut.io.wRawZero, false)
  for (ctl <- ref.controlSeq) {
    if (ctl.chunkStart) {
      poke(dut.io.mWarZero, true)
      for (i <- 0 until waitDelay) {
        step(1)
      }
      waitDelay += 2
      poke(dut.io.mWarZero, false)
    }
    expect(dut.io.control.wAddr, ctl.wAddr)
    expect(dut.io.control.mWe(0), ctl.mWeEven)
    expect(dut.io.control.mWe(1), ctl.mWeOdd)
    expect(dut.io.control.mRawInc, ctl.incMRaw)
    expect(dut.io.control.wWarInc, ctl.incWWar)
    expect(dut.io.mWarDec, ctl.decMWar)
    expect(dut.io.wRawDec, ctl.decWRaw)
    step(1)
  }
}

class W2mSequencerTestWRawZero(dut: W2mSequencer, val b: Int, val wMemSize: Int) extends PeekPokeTester(dut) {
  val ref = new DummyW2mControlSequencer(b, wMemSize)
  var waitDelay = 1
  poke(dut.io.mWarZero, false)
  for (ctl <- ref.controlSeq) {
    if (ctl.chunkStart) {
      poke(dut.io.wRawZero, true)
      for (i <- 0 until waitDelay) {
        step(1)
      }
      waitDelay += 2
      poke(dut.io.wRawZero, false)
    }
    expect(dut.io.control.wAddr, ctl.wAddr)
    expect(dut.io.control.mWe(0), ctl.mWeEven)
    expect(dut.io.control.mWe(1), ctl.mWeOdd)
    expect(dut.io.control.mRawInc, ctl.incMRaw)
    expect(dut.io.control.wWarInc, ctl.incWWar)
    expect(dut.io.mWarDec, ctl.decMWar)
    expect(dut.io.wRawDec, ctl.decWRaw)
    step(1)
  }
}

object W2mSequencerTests {
  def main(args: Array[String]): Unit = {
    val verilatorArgs = Array("--backend-name", "verilator", "--is-verbose", "true")
    val driverArgs = if (args.contains("verilator")) verilatorArgs else Array[String]()
    var ok = true
    val b = 2
    val memSize = 4
    val wAddrSize = Chisel.log2Up(memSize)
    ok &= Driver.execute(driverArgs, () => new W2mSequencer(b, wAddrSize))(dut => new W2mSequencerTestSequence(dut, b, memSize))
    ok &= Driver.execute(driverArgs, () => new W2mSequencer(b, wAddrSize))(dut => new W2mSequencerTestMWarZero(dut, b, memSize))
    ok &= Driver.execute(driverArgs, () => new W2mSequencer(b, wAddrSize))(dut => new W2mSequencerTestWRawZero(dut, b, memSize))
    if (!ok && args.contains("noexit"))
      System.exit(1)
  }
}
