# BxB
Using Chisel to prototype some aspects of TCA in RTL

---
# Name
**BxB** is named after the famous `B` parameter of Tomida-san's architecture

---
# Installation
The project uses Chisel to generate Verilog.

Chisel itself is just a library and will be fetched automatically by `sbt`,
but you need to install `java sdk`, `sbt` and `verilator` first.

Go through installation steps described [here](https://github.com/freechipsproject/chisel3#installation)
to have to have all dependencies installed.

# How to use it
Just run `sbt` in the project directory first
```
> cd bxb
> sbt
```
It will take some time for sbt to start and fetch all dependencies, when you do it first time

## Running tests
At the moment I do not have a method which invokes all the tests, but instead lots of test objects,
and each of the test objects contains the test for particular module.

Run `show test:discoveredMainClasses` to list all available test benches. You will see something like
```
sbt:bxb> show test:discoveredMainClasses
[info] Compiling 1 Scala source to /home/nez/devel/chisel/work/bxb/target/scala-2.11/test-classes ...
[info] Done compiling.
[info] * bxb.a2f.A2fPipelineTests
[info] * bxb.a2f.AccumulationPipelineTests
[info] * bxb.array.MacArrayTests
[info] * bxb.array.MacTests
[info] * bxb.memory.MemArrayTests
[info] * bxb.memory.TwoBlockRamTests
```

To run particular test bench run `test:runMain <name of test bench>`  
E.g.
```
> test:runMain bxb.a2f.A2fPipelineTests
```
will run test bench for `A2fPipeline`


## Give me Verilog
**DISCLAIMER**: eventually we are going to have toplevel module containing whole design with sensible parameters,
but it still is not there. What is here is lots of modules implementing different parts of the system, and you can
instantiate each of them using *dummy* parameters to see how it is going look.

Most of the modules implemented at the moment have associated object which instantiates them with some example parameters.
Typically I use `B=3`, `aWidth=2`, `fWidth=16`, just to see what the generator is going to produce.

You can list the modules using below command
```
sbt:bxb> show discoveredMainClasses
[info] * bxb.a2f.A2fPipeline
[info] * bxb.a2f.AccumulationPipeline
[info] * bxb.a2f.AddressPipeline
[info] * bxb.array.Mac
[info] * bxb.array.MacArray
[info] * bxb.memory.BlockRam
[info] * bxb.memory.MemArray
[info] * bxb.memory.TwoBlockMemArray
[info] * bxb.memory.TwoBlockRam
```

To see what verilog (with example parameters) of particular module will look like, run `test:runMain <name of the module>`
E.g.
```
> test:runMain bxb.array.MacArray
```
will give you Verilog for systolic array (`B=3`) on standart output, at the same time it will produce `MacArray.v` file
which you can examine with your favorite text editor.

To change generation parameters, open the file containing module you are going to instantiate, and change parameters
of the module passed to `genVerilog()` method, to ones you like.

E.g. changing
```
object MacArray {
...
  def main(args: Array[String]): Unit = {
    println(getVerilog(new MacArray(3, 16, 2)))
  }
}
```
to
```
object MacArray {
...
  def main(args: Array[String]): Unit = {
    println(getVerilog(new MacArray(32, 16, 2)))
  }
}
```
will give you a systolic array of size `B=32` instead of `3`

---
# Modules

## TwoBlockRam
![TwoBlockRam](/docs/diagrams/TwoBlockRam.png)

## Accumulation Pipeline

**one stage block**
![TwoBlockRam](/docs/diagrams/accumulationPipelineElement.png)

**assembled pipeline (B=3)**
![TwoBlockRam](/docs/diagrams/accumulationPipeline.png)

**assembled a2f pipeline (B=3)**
![a2fPipeline](/docs/diagrams/a2fPipeline.png)

**assembled w2m pipeline (B=3)**
![w2mPipeline](/docs/diagrams/w2mPipeline.png)

**A2F <-> W2M sync**
![w2m_a2f_sync.png](/docs/diagrams/w2m_a2f_sync.png)
