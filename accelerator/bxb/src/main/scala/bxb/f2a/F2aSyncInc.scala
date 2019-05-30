package bxb.f2a

import chisel3._

class F2aSyncInc extends Bundle {
  val qWar = Bool()
  val fWar = Bool()
  val aRaw = Bool()
}

object F2aSyncInc {
  def apply() = {
    new F2aSyncInc
  }
}
