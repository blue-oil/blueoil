package bxb

import chisel3._
import chisel3.util._

import bxb.a2f.{A2f, A2fParameters}
import bxb.adma.{ADma, ADmaParameters}
import bxb.array.{MacArray}
import bxb.avalon.{ReadMasterIO, WriteMasterIO, SlaveIO}
import bxb.f2a.{F2a, F2aParameters}
import bxb.fdma.{FDma, FDmaParameters}
import bxb.rdma.{RDma}
import bxb.w2m.{W2m}
import bxb.wqdma.{WDma, QDma}
import bxb.sync.{SemaphorePair, ConsumerSyncMux}
import bxb.memory.{MemArray, PackedBlockRam, TwoBlockMemArray}

// FIXME: not the best way of it
object BxbCsrField {
  // parameter registers
  val start = 0
  val admaInputAddress = 1
  val admaInputHCount = 2
  val admaInputWCount = 3
  val admaInputCCount = 4
  val admaTopTileH = 5
  val admaMiddleTileH = 6
  val admaBottomTileH = 7
  val admaLeftTileW = 8
  val admaMiddleTileW = 9
  val admaRightTileW = 10
  val admaLeftRowToRowDistance = 11
  val admaMiddleRowToRowDistance = 12
  val admaRightRowToRowDistance = 13
  val admaLeftStep = 14
  val admaMiddleStep = 15
  val admaTopRowDistance = 16
  val admaMidRowDistance = 17
  val admaInputSpace = 18
  val admaTopBottomLeftPad = 19
  val admaTopBottomMiddlePad = 20
  val admaTopBottomRightPad = 21
  val admaSidePad = 22
  val wdmaStartAddress = 23
  val wdmaOutputHCount = 24
  val wdmaOutputWCount = 25
  val wdmaKernelBlockCount = 26
  val fdmaOutputAddress = 27
  val fdmaOutputHCount = 28
  val fdmaOutputWCount = 29
  val fdmaOutputCCount = 30
  val fdmaRegularTileH = 31
  val fdmaLastTileH = 32
  val fdmaRegularTileW = 33
  val fdmaLastTileW = 34
  val fdmaRegularRowToRowDistance = 35
  val fdmaLastRowToRowDistance = 36
  val fdmaOutputSpace = 37
  val fdmaRowDistance = 38
  val a2fInputCCount = 39
  val a2fKernelVCount = 40
  val a2fKernelHCount = 41
  val a2fTileStep = 42
  val a2fTileGap = 43
  val a2fOutputHCount = 44
  val a2fOutputWCount = 45
  val a2fRegularTileH = 46
  val a2fLastTileH = 47
  val a2fRegularTileW = 48
  val a2fLastTileW = 49
  val qdmaStartAddress = 50
  val bnqEnable = 51

  val parameterCount = 52

  // status registers
  val statusRegister = parameterCount + 0
}

class BxbCsr(avalonAddrWidth: Int, tileCountWidth: Int) extends Module {
  // FIXME: rid of copypaste
  val hCountWidth = 6
  val wCountWidth = 6
  val blockCountWidth = 14
  val io = IO(new Bundle {
    // Avalon slave interface
    val avalonSlave = SlaveIO(Chisel.log2Up(50), 32) // FIXME:rid of log2Up(50)

    // TODO: parameter & start slave
    val start = Output(Bool())

    // ADMA parameters
    val admaParameters = Output(ADmaParameters(avalonAddrWidth, tileCountWidth))

    // WDMA parameters
    val wdmaStartAddress = Output(UInt(avalonAddrWidth.W))
    val wdmaOutputHCount = Output(UInt(hCountWidth.W))
    val wdmaOutputWCount = Output(UInt(wCountWidth.W))
    val wdmaKernelBlockCount = Output(UInt(blockCountWidth.W))

    // WDMA parameters
    val qdmaStartAddress = Output(UInt(avalonAddrWidth.W))
    val qdmaOutputHCount = Output(UInt(hCountWidth.W))
    val qdmaOutputWCount = Output(UInt(wCountWidth.W))
    val qdmaKernelBlockCount = Output(UInt(blockCountWidth.W))

    // FDMA parameters
    val fdmaParameters = Output(FDmaParameters(avalonAddrWidth, tileCountWidth))

    val rdmaOutputAddress = Output(UInt(avalonAddrWidth.W))
    val rdmaOutputHCount = Output(UInt(6.W))
    val rdmaOutputWCount = Output(UInt(6.W))
    val rdmaOutputCCount = Output(UInt(6.W))
    val rdmaRegularTileH = Output(UInt(tileCountWidth.W))
    val rdmaLastTileH = Output(UInt(tileCountWidth.W))
    val rdmaRegularTileW = Output(UInt(tileCountWidth.W))
    val rdmaLastTileW = Output(UInt(tileCountWidth.W))
    val rdmaRegularRowToRowDistance = Output(UInt(tileCountWidth.W))
    val rdmaLastRowToRowDistance = Output(UInt(tileCountWidth.W))
    val rdmaOutputSpace = Output(UInt(avalonAddrWidth.W))
    val rdmaRowDistance = Output(UInt(avalonAddrWidth.W))

    // A2F parameters
    val a2fParameters = Output(A2fParameters(tileCountWidth))

    // F2A parameters
    val f2rParameters = Output(F2aParameters(tileCountWidth))

    val bnqEnable = Output(Bool())

    // Modules Status
    val a2fStatusReady = Input(Bool())
    val admaStatusReady = Input(Bool())
    val wdmaStatusReady = Input(Bool())
    val fdmaStatusReady = Input(Bool())
    val qdmaStatusReady = Input(Bool())
    val f2rStatusReady = Input(Bool())
    val rdmaStatusReady = Input(Bool())
  })

  val field = RegInit(VecInit(Seq.fill(BxbCsrField.parameterCount){0.U(32.W)}))
  when(io.avalonSlave.write & (io.avalonSlave.address =/= BxbCsrField.start.U)) {
    field(io.avalonSlave.address) := io.avalonSlave.writeData
  }

  val readData = Reg(UInt(32.W))
  val status = Cat(0.U(25.W),
    /* 6 */ io.rdmaStatusReady,
    /* 5 */ io.f2rStatusReady,
    /* 4 */ io.qdmaStatusReady,
    /* 3 */ io.a2fStatusReady,
    /* 2 */ io.admaStatusReady,
    /* 1 */ io.wdmaStatusReady,
    /* 0 */ io.fdmaStatusReady
  )
  when(io.avalonSlave.read & (io.avalonSlave.address =/= BxbCsrField.statusRegister.U)) {
    readData := field(io.avalonSlave.address)
  }.otherwise {
    readData := status
  }
  io.avalonSlave.readData := readData

  io.start := RegNext(io.avalonSlave.write & (io.avalonSlave.address === BxbCsrField.start.U))
  // ADMA
  io.admaParameters.inputAddress := field(BxbCsrField.admaInputAddress.U)
  io.admaParameters.inputHCount := field(BxbCsrField.admaInputHCount.U)
  io.admaParameters.inputWCount := field(BxbCsrField.admaInputWCount.U)
  io.admaParameters.inputCCount := field(BxbCsrField.admaInputCCount.U)
  io.admaParameters.outputCCount := field(BxbCsrField.fdmaOutputCCount.U)
  io.admaParameters.topTileH := field(BxbCsrField.admaTopTileH.U)
  io.admaParameters.middleTileH := field(BxbCsrField.admaMiddleTileH.U)
  io.admaParameters.bottomTileH := field(BxbCsrField.admaBottomTileH.U)
  io.admaParameters.leftTileW := field(BxbCsrField.admaLeftTileW.U)
  io.admaParameters.middleTileW := field(BxbCsrField.admaMiddleTileW.U)
  io.admaParameters.rightTileW := field(BxbCsrField.admaRightTileW.U)
  io.admaParameters.leftRowToRowDistance := field(BxbCsrField.admaLeftRowToRowDistance.U)
  io.admaParameters.middleRowToRowDistance := field(BxbCsrField.admaMiddleRowToRowDistance.U)
  io.admaParameters.rightRowToRowDistance := field(BxbCsrField.admaRightRowToRowDistance.U)
  io.admaParameters.leftStep := field(BxbCsrField.admaLeftStep.U)
  io.admaParameters.middleStep := field(BxbCsrField.admaMiddleStep.U)
  io.admaParameters.topRowDistance := field(BxbCsrField.admaTopRowDistance.U)
  io.admaParameters.midRowDistance := field(BxbCsrField.admaMidRowDistance.U)
  io.admaParameters.inputSpace := field(BxbCsrField.admaInputSpace.U)
  io.admaParameters.topBottomLeftPad := field(BxbCsrField.admaTopBottomLeftPad.U)
  io.admaParameters.topBottomMiddlePad := field(BxbCsrField.admaTopBottomMiddlePad.U)
  io.admaParameters.topBottomRightPad := field(BxbCsrField.admaTopBottomRightPad.U)
  io.admaParameters.sidePad := field(BxbCsrField.admaSidePad.U)
  // WDMA
  io.wdmaStartAddress := field(BxbCsrField.wdmaStartAddress.U)
  io.wdmaOutputHCount := field(BxbCsrField.wdmaOutputHCount.U)
  io.wdmaOutputWCount := field(BxbCsrField.wdmaOutputWCount.U)
  io.wdmaKernelBlockCount := field(BxbCsrField.wdmaKernelBlockCount.U)
  // QDMA
  io.qdmaStartAddress := field(BxbCsrField.qdmaStartAddress.U)
  io.qdmaOutputHCount := field(BxbCsrField.wdmaOutputHCount.U)
  io.qdmaOutputWCount := field(BxbCsrField.wdmaOutputWCount.U)
  io.qdmaKernelBlockCount := field(BxbCsrField.fdmaOutputCCount.U)
  // FDMA
  io.fdmaParameters.outputAddress := field(BxbCsrField.fdmaOutputAddress.U)
  io.fdmaParameters.outputHCount := field(BxbCsrField.fdmaOutputHCount.U)
  io.fdmaParameters.outputWCount := field(BxbCsrField.fdmaOutputWCount.U)
  io.fdmaParameters.outputCCount := field(BxbCsrField.fdmaOutputCCount.U)
  io.fdmaParameters.regularTileH := field(BxbCsrField.fdmaRegularTileH.U)
  io.fdmaParameters.lastTileH := field(BxbCsrField.fdmaLastTileH.U)
  io.fdmaParameters.regularTileW := field(BxbCsrField.fdmaRegularTileW.U)
  io.fdmaParameters.lastTileW := field(BxbCsrField.fdmaLastTileW.U)
  io.fdmaParameters.regularRowToRowDistance := field(BxbCsrField.fdmaRegularRowToRowDistance.U)
  io.fdmaParameters.lastRowToRowDistance := field(BxbCsrField.fdmaLastRowToRowDistance.U)
  io.fdmaParameters.outputSpace := field(BxbCsrField.fdmaOutputSpace.U)
  io.fdmaParameters.rowDistance := field(BxbCsrField.fdmaRowDistance.U)
  // RDMA
  io.rdmaOutputAddress := field(BxbCsrField.fdmaOutputAddress.U)
  io.rdmaOutputHCount := field(BxbCsrField.fdmaOutputHCount.U)
  io.rdmaOutputWCount := field(BxbCsrField.fdmaOutputWCount.U)
  io.rdmaOutputCCount := field(BxbCsrField.fdmaOutputCCount.U)
  io.rdmaRegularTileH := field(BxbCsrField.fdmaRegularTileH.U)
  io.rdmaLastTileH := field(BxbCsrField.fdmaLastTileH.U)
  io.rdmaRegularTileW := field(BxbCsrField.fdmaRegularTileW.U)
  io.rdmaLastTileW := field(BxbCsrField.fdmaLastTileW.U)
  io.rdmaRegularRowToRowDistance := field(BxbCsrField.fdmaRegularRowToRowDistance.U)
  io.rdmaLastRowToRowDistance := field(BxbCsrField.fdmaLastRowToRowDistance.U)
  io.rdmaOutputSpace := field(BxbCsrField.fdmaOutputSpace.U)
  io.rdmaRowDistance := field(BxbCsrField.fdmaRowDistance.U)
  // A2F
  io.a2fParameters.inputCCount := field(BxbCsrField.a2fInputCCount.U)
  io.a2fParameters.kernelVCount := field(BxbCsrField.a2fKernelVCount.U)
  io.a2fParameters.kernelHCount := field(BxbCsrField.a2fKernelHCount.U)
  io.a2fParameters.tileStep := field(BxbCsrField.a2fTileStep.U)
  io.a2fParameters.tileGap := field(BxbCsrField.a2fTileGap.U)
  io.a2fParameters.outputHCount := field(BxbCsrField.a2fOutputHCount.U)
  io.a2fParameters.outputWCount := field(BxbCsrField.a2fOutputWCount.U)
  io.a2fParameters.outputCCount := field(BxbCsrField.fdmaOutputCCount.U)
  io.a2fParameters.regularTileH := field(BxbCsrField.a2fRegularTileH.U)
  io.a2fParameters.lastTileH := field(BxbCsrField.a2fLastTileH.U)
  io.a2fParameters.regularTileW := field(BxbCsrField.a2fRegularTileW.U)
  io.a2fParameters.lastTileW := field(BxbCsrField.a2fLastTileW.U)
  // F2R
  io.f2rParameters.outputHCount := field(BxbCsrField.a2fOutputHCount.U)
  io.f2rParameters.outputWCount := field(BxbCsrField.a2fOutputWCount.U)
  io.f2rParameters.outputCCount := field(BxbCsrField.fdmaOutputCCount.U)
  io.f2rParameters.regularTileH := field(BxbCsrField.a2fRegularTileH.U)
  io.f2rParameters.lastTileH := field(BxbCsrField.a2fLastTileH.U)
  io.f2rParameters.regularTileW := field(BxbCsrField.a2fRegularTileW.U)
  io.f2rParameters.lastTileW := field(BxbCsrField.a2fLastTileW.U)
  // BNQ Enable
  io.bnqEnable := field(BxbCsrField.bnqEnable)(0)
}

class Bxb(dataMemSize: Int, wmemSize: Int, qmemSize: Int) extends Module {
  require(isPow2(dataMemSize))
  require(isPow2(wmemSize))

  val b = 32
  val aWidth = 2
  val fWidth = 16
  val wWidth = 1

  val wEntriesCount = wmemSize / b
  val wSemaWidth = Chisel.log2Floor(wEntriesCount) + 1

  val qEntriesCount = qmemSize
  val qSemaWidth = Chisel.log2Floor(qEntriesCount) + 1

  val avalonAddrWidth = 32
  val maxBurst = 32

  val dataAddrWidth = Chisel.log2Ceil(dataMemSize)
  val wAddrWidth = Chisel.log2Ceil(wmemSize)
  val qAddrWidth = Chisel.log2Ceil(qmemSize)
  val tileCountWidth = dataAddrWidth

  // FIXME: rid of copypaste
  val hCountWidth = 6
  val wCountWidth = 6
  val blockCountWidth = 14

  val io = IO(new Bundle {
    // Avalon slave interface
    val csrSlave = SlaveIO(Chisel.log2Up(50), 32) // FIXME:rid of log2Up(50)

    // ADMA Avalon Interface
    val admaAvalon = ReadMasterIO(avalonAddrWidth, b * aWidth)

    // WDMA Avalon Interface
    val wdmaAvalon = ReadMasterIO(avalonAddrWidth, b * wWidth)

    // QDMA Avalon Interface
    val qdmaAvalon = ReadMasterIO(avalonAddrWidth, 64)

    // FDMA Avalon Interface
    val fdmaAvalon = WriteMasterIO(avalonAddrWidth, 128)

    // RDMA Avalon Interface
    val rdmaAvalon = WriteMasterIO(avalonAddrWidth, (b * aWidth))
  })

  val csr = Module(new BxbCsr(avalonAddrWidth, tileCountWidth))
  io.csrSlave <> csr.io.avalonSlave

  val asema = Module(new SemaphorePair(2, 0, 2))
  val amem = Module(new MemArray(b, dataMemSize, aWidth))

  val wsema = Module(new SemaphorePair(wSemaWidth, 0, wEntriesCount))
  val wmem = Module(new PackedBlockRam(b, wmemSize, wWidth))

  val msema = Module(new SemaphorePair(2, 0, 2))
  val macArray = Module(new MacArray(b, fWidth, aWidth))

  val fsemaConsumerMux = Module(new ConsumerSyncMux)
  fsemaConsumerMux.io.select := csr.io.bnqEnable
  val fsema = Module(new SemaphorePair(2, 0, 2))
  fsema.io.consumer <> fsemaConsumerMux.io.out

  val fmem = Module(new TwoBlockMemArray(b, dataMemSize, fWidth))
  val fdmaRead = Wire(fmem.io.readB.cloneType)
  val f2rRead = Wire(fmem.io.readB.cloneType)
  fmem.io.readB := Mux(csr.io.bnqEnable, f2rRead, fdmaRead) 

  val qsema = Module(new SemaphorePair(qSemaWidth, 0, qEntriesCount))
  val qmem = Module(new PackedBlockRam(b, qmemSize, 40))

  val rsema = Module(new SemaphorePair(2, 0, 2))
  val rmem = Module(new MemArray(b, dataMemSize, aWidth))

  val adma = Module(new ADma(b, dataAddrWidth, avalonAddrWidth, maxBurst))
  adma.io.start := csr.io.start
  asema.io.producer <> adma.io.aSync
  io.admaAvalon <> adma.io.avalonMaster

  amem.io.write := adma.io.amemWrite

  adma.io.parameters := csr.io.admaParameters
  csr.io.admaStatusReady := adma.io.statusReady

  val wdma = Module(new WDma(b, avalonAddrWidth, b * wWidth, wAddrWidth))
  wdma.io.start := csr.io.start
  wsema.io.producer <> wdma.io.wSync
  io.wdmaAvalon <> wdma.io.avalonMaster

  // FIXME: refactor parameters
  wdma.io.startAddress := csr.io.wdmaStartAddress
  wdma.io.outputHCount := csr.io.wdmaOutputHCount
  wdma.io.outputWCount := csr.io.wdmaOutputWCount
  wdma.io.kernelBlockCount := csr.io.wdmaKernelBlockCount

  // FIXME: refactor memory interface
  wmem.io.write := wdma.io.wmemWrite
  csr.io.wdmaStatusReady := wdma.io.statusReady

  val fdma = Module(new FDma(b, dataAddrWidth, avalonAddrWidth, 128, maxBurst))
  fdma.io.start := ~csr.io.bnqEnable & csr.io.start
  fsemaConsumerMux.io.b <> fdma.io.fSync
  io.fdmaAvalon <> fdma.io.avalonMaster

  // FIXME: refactor memory interface
  fdmaRead := fdma.io.fmemRead
  fdma.io.fmemQ := fmem.io.qB

  fdma.io.parameters := csr.io.fdmaParameters
  csr.io.fdmaStatusReady := fdma.io.statusReady

  val rdma = Module(new RDma(b, dataAddrWidth, avalonAddrWidth, maxBurst))
  rdma.io.start := csr.io.bnqEnable & csr.io.start
  rsema.io.consumer <> rdma.io.rSync
  io.rdmaAvalon <> rdma.io.avalonMaster

  // FIXME: refactor parameters
  rdma.io.outputAddress := csr.io.rdmaOutputAddress
  rdma.io.outputHCount := csr.io.rdmaOutputHCount
  rdma.io.outputWCount := csr.io.rdmaOutputWCount
  rdma.io.outputCCount := csr.io.rdmaOutputCCount
  rdma.io.regularTileH := csr.io.rdmaRegularTileH
  rdma.io.lastTileH := csr.io.rdmaLastTileH
  rdma.io.regularTileW := csr.io.rdmaRegularTileW
  rdma.io.lastTileW := csr.io.rdmaLastTileW
  rdma.io.regularRowToRowDistance := csr.io.rdmaRegularRowToRowDistance
  rdma.io.lastRowToRowDistance := csr.io.rdmaLastRowToRowDistance
  rdma.io.outputSpace := csr.io.rdmaOutputSpace
  rdma.io.rowDistance := csr.io.rdmaRowDistance

  // FIXME: refactor memory interface
  rmem.io.read := rdma.io.rmemRead
  rdma.io.rmemQ := rmem.io.q
  csr.io.rdmaStatusReady := rdma.io.statusReady

  val qdma = Module(new QDma(b, avalonAddrWidth, 64, qAddrWidth))
  qdma.io.start := csr.io.bnqEnable & csr.io.start
  qsema.io.producer <> qdma.io.qSync
  io.qdmaAvalon <> qdma.io.avalonMaster

  // FIXME: refactor parameters
  qdma.io.startAddress := csr.io.qdmaStartAddress
  qdma.io.outputHCount := csr.io.qdmaOutputHCount
  qdma.io.outputWCount := csr.io.qdmaOutputWCount
  qdma.io.kernelBlockCount := csr.io.qdmaKernelBlockCount

  qmem.io.write := qdma.io.qmemWrite
  csr.io.qdmaStatusReady := qdma.io.statusReady

  val w2m = Module(new W2m(b, wmemSize))
  wsema.io.consumer <> w2m.io.wSync
  // FIXME: refactor memory interface
  wmem.io.read := w2m.io.wmemRead
  w2m.io.wmemQ := wmem.io.q

  msema.io.producer <> w2m.io.mSync

  macArray.io.mIn := w2m.io.mOut
  macArray.io.mWe := w2m.io.mWe

  val a2f = Module(new A2f(b, dataMemSize, aWidth, fWidth))
  a2f.io.start := csr.io.start

  asema.io.consumer <> a2f.io.aSync
  // FIXME: refactor memory interface
  amem.io.read := a2f.io.amemRead
  a2f.io.amemQ := amem.io.q

  msema.io.consumer <> a2f.io.mSync
  // FIXME: refactor mac interface
  macArray.io.aIn := a2f.io.aOut
  macArray.io.evenOddIn := a2f.io.evenOddOut
  a2f.io.accIn := macArray.io.accOut

  fsema.io.producer <> a2f.io.fSync
  // FIXME: refactor memory interface
  fmem.io.readA := a2f.io.fmemRead
  fmem.io.writeA := a2f.io.fmemWrite
  a2f.io.fmemQ := fmem.io.qA

  a2f.io.parameters := csr.io.a2fParameters
  csr.io.a2fStatusReady := a2f.io.statusReady

  val f2r = Module(new F2a(b, dataMemSize, qmemSize, aWidth, fWidth))
  f2r.io.start := csr.io.bnqEnable & csr.io.start

  rmem.io.write := f2r.io.amemWrite

  f2rRead := f2r.io.fmemRead
  f2r.io.fmemQ := fmem.io.qB

  qmem.io.read := f2r.io.qmemRead
  f2r.io.qmemQ := qmem.io.q

  rsema.io.producer <> f2r.io.aSync
  qsema.io.consumer <> f2r.io.qSync
  fsemaConsumerMux.io.a <> f2r.io.fSync

  f2r.io.parameters := csr.io.f2rParameters
  csr.io.f2rStatusReady := f2r.io.statusReady
}

object Bxb {
  def main(args: Array[String]): Unit = {
    chisel3.Driver.execute(args, () => new Bxb(4 * 1024, 1024, 32))
  }
}
