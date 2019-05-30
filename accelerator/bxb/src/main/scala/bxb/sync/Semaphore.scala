package bxb.sync

import chisel3._

import bxb.util.{Util}

class Semaphore(counterWidth: Int, initValue: Int) extends Module {
  val io = IO(new Bundle {
    // decrement request
    val decrement = Input(Bool())
    // back pressure for decrement request
    // decrement request will be served if zero != 0 and will not be served otherwise
    val zero = Output(Bool())
    // increment request
    val increment = Input(Bool())
    // back pressure for increment request
    val full = Output(Bool())
  })

  val maxCount = ((0x1 << counterWidth) - 1).U
  val count = RegInit(initValue.U(counterWidth.W))
  val zero = (count === 0.U)
  val full = (count === maxCount)
  // when decrement and increment asserted simultaneously the state doesn't change
  // unless the counter is already full or zero
  //
  // when counter is zero and both request signals asserted
  // counter should be incremented first and allow decrementing side to decrement it in the next cycle
  //
  // when counter is full and both request signals asserted
  // counter should be decremented first and allow incrementing side to increment it in the next cycle
  when(io.decrement & ((~io.increment & ~zero) | (io.increment & full))) {
    count := count - 1.U
  }.elsewhen(io.increment & ((~io.decrement & ~full) | (io.decrement & zero))) {
    count := count + 1.U
  }
  io.zero := zero
  io.full := full
}

object Semaphore {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new Semaphore(2, 0)))
  }
}
