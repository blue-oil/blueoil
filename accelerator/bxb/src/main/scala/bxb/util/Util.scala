package bxb.util

import chisel3._
import chisel3.core.{UserModule}
import firrtl._

object Util {
  def getVerilog(dut: => UserModule): String = {
    return chisel3.Driver.execute(Array[String](), {() => dut}) match {
      case s:ChiselExecutionSuccess => s.firrtlResultOption match {
        case Some(f:FirrtlExecutionSuccess) => f.emitted
      }
    }
  }
}
