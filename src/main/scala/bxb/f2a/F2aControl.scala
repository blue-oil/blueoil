package bxb.f2a

import chisel3._

class F2aControl() extends Bundle {
  val qWe = Bool()
  val syncInc = F2aSyncInc()
}

object F2aControl {
  def apply() = {
    new F2aControl()
  }
}
