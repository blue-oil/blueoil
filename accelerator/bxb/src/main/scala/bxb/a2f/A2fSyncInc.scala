package bxb.a2f

import chisel3._

class A2fSyncInc extends Bundle {
  val aWar = Bool()
  val mWar = Bool()
  val fRaw = Bool()
}

object A2fSyncInc {
  def apply() = {
    new A2fSyncInc
  }
}
