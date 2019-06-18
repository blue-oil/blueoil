package bxb.memory

import chisel3._

import bxb.util.{Util}

class TwoBlockRam(size: Int, dataWidth: Int) extends Module {
  val addrWidth = Chisel.log2Up(size)
  val addrMsb = addrWidth - 1
  val bankAddrMsb = addrWidth - 2
  val io = IO(new Bundle {
    val readA = Input(ReadPort(addrWidth))
    val writeA = Input(WritePort(addrWidth, dataWidth))
    val qA = Output(UInt(dataWidth.W))
    val readB = Input(ReadPort(addrWidth))
    val qB = Output(UInt(dataWidth.W))
  })
  val banks = Seq.fill(2){Module(new BlockRam(size / 2, dataWidth))}
  val readBankA = io.readA.addr(addrMsb)
  val readReq0A = (readBankA === 0.U) & io.readA.enable
  val readBankAddrA = io.readA.addr(bankAddrMsb, 0)
  val readBankB = io.readB.addr(addrMsb)
  val readReq1B = (readBankB === 1.U) & io.readB.enable
  val readBankAddrB = io.readB.addr(bankAddrMsb, 0)
  banks(0).io.read.addr := Mux(readReq0A, readBankAddrA, readBankAddrB)
  banks(0).io.read.enable := true.B
  io.qA := Mux(RegNext(readReq0A), banks(0).io.readQ, banks(1).io.readQ)
  banks(1).io.read.addr := Mux(readReq1B, readBankAddrB, readBankAddrA)
  banks(1).io.read.enable := true.B
  io.qB := Mux(RegNext(readReq1B), banks(1).io.readQ, banks(0).io.readQ)
  val writeBankA = io.writeA.addr(addrMsb)
  val writeReq0A = (writeBankA === 0.U) & io.writeA.enable
  val writeReq1A = (writeBankA === 1.U) & io.writeA.enable
  banks(0).io.write.addr := io.writeA.addr
  banks(0).io.write.data := io.writeA.data
  banks(0).io.write.enable := writeReq0A
  banks(1).io.write.addr := io.writeA.addr
  banks(1).io.write.enable := writeReq1A
  banks(1).io.write.data := io.writeA.data
}

object TwoBlockRam {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new TwoBlockRam(4096 * 2, 2)))
  }
}
