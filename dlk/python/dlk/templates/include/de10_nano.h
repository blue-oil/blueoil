/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include "global.h"
#include "memdriver.h"
#include "time_measurement.h"
#include <cassert>

namespace de10_nano {

//
// TCA
//
struct Csr {
    static constexpr uint32_t start = 0;
    static constexpr uint32_t admaInputAddress = 1;
    static constexpr uint32_t admaInputHCount = 2;
    static constexpr uint32_t admaInputWCount = 3;
    static constexpr uint32_t admaInputCCount = 4;
    static constexpr uint32_t admaTopTileH = 5;
    static constexpr uint32_t admaMiddleTileH = 6;
    static constexpr uint32_t admaBottomTileH = 7;
    static constexpr uint32_t admaLeftTileW = 8;
    static constexpr uint32_t admaMiddleTileW = 9;
    static constexpr uint32_t admaRightTileW = 10;
    static constexpr uint32_t admaLeftRowToRowDistance = 11;
    static constexpr uint32_t admaMiddleRowToRowDistance = 12;
    static constexpr uint32_t admaRightRowToRowDistance = 13;
    static constexpr uint32_t admaLeftStep = 14;
    static constexpr uint32_t admaMiddleStep = 15;
    static constexpr uint32_t admaTopRowDistance = 16;
    static constexpr uint32_t admaMidRowDistance = 17;
    static constexpr uint32_t admaInputSpace = 18;
    static constexpr uint32_t admaTopBottomLeftPad = 19;
    static constexpr uint32_t admaTopBottomMiddlePad = 20;
    static constexpr uint32_t admaTopBottomRightPad = 21;
    static constexpr uint32_t admaSidePad = 22;
    static constexpr uint32_t wdmaStartAddress = 23;
    static constexpr uint32_t wdmaOutputHCount = 24;
    static constexpr uint32_t wdmaOutputWCount = 25;
    static constexpr uint32_t wdmaKernelBlockCount = 26;
    static constexpr uint32_t fdmaOutputAddress = 27;
    static constexpr uint32_t fdmaOutputHCount = 28;
    static constexpr uint32_t fdmaOutputWCount = 29;
    static constexpr uint32_t fdmaOutputCCount = 30;
    static constexpr uint32_t fdmaRegularTileH = 31;
    static constexpr uint32_t fdmaLastTileH = 32;
    static constexpr uint32_t fdmaRegularTileW = 33;
    static constexpr uint32_t fdmaLastTileW = 34;
    static constexpr uint32_t fdmaRegularRowToRowDistance = 35;
    static constexpr uint32_t fdmaLastRowToRowDistance = 36;
    static constexpr uint32_t fdmaOutputSpace = 37;
    static constexpr uint32_t fdmaRowDistance = 38;
    static constexpr uint32_t a2fInputCCount = 39;
    static constexpr uint32_t a2fKernelVCount = 40;
    static constexpr uint32_t a2fKernelHCount = 41;
    static constexpr uint32_t a2fTileStep = 42;
    static constexpr uint32_t a2fTileGap = 43;
    static constexpr uint32_t a2fOutputHCount = 44;
    static constexpr uint32_t a2fOutputWCount = 45;
    static constexpr uint32_t a2fRegularTileH = 46;
    static constexpr uint32_t a2fLastTileH = 47;
    static constexpr uint32_t a2fRegularTileW = 48;
    static constexpr uint32_t a2fLastTileW = 49;
    static constexpr uint32_t qdmaStartAddress = 50;
    static constexpr uint32_t bnqEnable = 51;

    static constexpr uint32_t statusRegister = 52;
};

struct Parameters {
    uint32_t admaInputAddress;
    uint32_t admaInputHCount;
    uint32_t admaInputWCount;
    uint32_t admaInputCCount;
    uint32_t admaTopTileH;
    uint32_t admaMiddleTileH;
    uint32_t admaBottomTileH;
    uint32_t admaLeftTileW;
    uint32_t admaMiddleTileW;
    uint32_t admaRightTileW;
    uint32_t admaLeftRowToRowDistance;
    uint32_t admaMiddleRowToRowDistance;
    uint32_t admaRightRowToRowDistance;
    uint32_t admaLeftStep;
    uint32_t admaMiddleStep;
    uint32_t admaTopRowDistance;
    uint32_t admaMidRowDistance;
    uint32_t admaInputSpace;
    uint32_t admaTopBottomLeftPad;
    uint32_t admaTopBottomMiddlePad;
    uint32_t admaTopBottomRightPad;
    uint32_t admaSidePad;
    uint32_t wdmaStartAddress;
    uint32_t wdmaOutputHCount;
    uint32_t wdmaOutputWCount;
    uint32_t wdmaKernelBlockCount;
    uint32_t fdmaOutputAddress;
    uint32_t fdmaOutputHCount;
    uint32_t fdmaOutputWCount;
    uint32_t fdmaOutputCCount;
    uint32_t fdmaRegularTileH;
    uint32_t fdmaLastTileH;
    uint32_t fdmaRegularTileW;
    uint32_t fdmaLastTileW;
    uint32_t fdmaRegularRowToRowDistance;
    uint32_t fdmaLastRowToRowDistance;
    uint32_t fdmaOutputSpace;
    uint32_t fdmaRowDistance;
    uint32_t a2fInputCCount;
    uint32_t a2fKernelVCount;
    uint32_t a2fKernelHCount;
    uint32_t a2fTileStep;
    uint32_t a2fTileGap;
    uint32_t a2fOutputHCount;
    uint32_t a2fOutputWCount;
    uint32_t a2fRegularTileH;
    uint32_t a2fLastTileH;
    uint32_t a2fRegularTileW;
    uint32_t a2fLastTileW;
    uint32_t qdmaStartAddress;
    uint32_t bnqEnable;
};

Parameters calcParameters(uint32_t inputHeight, uint32_t inputWidth, uint32_t inputChannels, uint32_t inputTileWidth, uint32_t inputTileHeight,
    uint32_t outputChannels, uint32_t kernelHeight, uint32_t kernelWidth, uint32_t inputAddress, uint32_t kernelAddress, uint32_t thresholdAddress, uint32_t outputAddress, bool enable_bnq) {

  auto divRoundUp = [](uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
  };

  constexpr uint32_t maxBurst = 32;
  constexpr uint32_t b = 32;

  assert((kernelHeight == 3 && kernelWidth == 3) || (kernelHeight == 1 && kernelWidth == 1));

  uint32_t pad = (kernelHeight == 1) ? 0 : 1;
  uint32_t dep = kernelHeight - 1;

  auto outputHeight = inputHeight + 2 * pad - dep;
  auto outputWidth = inputWidth + 2 * pad - dep;

  auto outputTileHeight = inputTileHeight - dep;
  auto outputTileWidth = inputTileWidth - dep;

  assert(inputTileHeight > dep && inputTileWidth > dep);

  auto hCount = divRoundUp(outputHeight, outputTileHeight);
  auto wCount = divRoundUp(outputWidth, outputTileWidth);

  // ADMA Parameters
  Parameters p;
  p.admaInputAddress = inputAddress;
  p.admaInputHCount = hCount;
  p.admaInputWCount = wCount;

  p.admaInputCCount = divRoundUp(inputChannels, b);

  p.admaTopTileH = (hCount == 1) ? inputHeight : (inputTileHeight - pad);
  p.admaMiddleTileH = inputTileHeight;
  p.admaBottomTileH = inputHeight + pad - (hCount - 1)  * (inputTileHeight - dep);

  p.admaLeftTileW = (wCount == 1) ? inputWidth : (inputTileWidth - pad);
  p.admaMiddleTileW = inputTileWidth;
  p.admaRightTileW = (wCount == 1) ? inputWidth : inputWidth + pad - (wCount - 1) * (inputTileWidth - dep);

  p.admaLeftRowToRowDistance = inputWidth - p.admaLeftTileW + ((p.admaLeftTileW % maxBurst == 0) ? maxBurst : p.admaLeftTileW % maxBurst);
  p.admaMiddleRowToRowDistance = inputWidth - p.admaMiddleTileW + ((p.admaMiddleTileW % maxBurst == 0) ? maxBurst : p.admaMiddleTileW % maxBurst);
  p.admaRightRowToRowDistance = inputWidth - p.admaRightTileW + ((p.admaRightTileW % maxBurst == 0) ? maxBurst : p.admaRightTileW % maxBurst);

  p.admaLeftStep = p.admaLeftTileW - dep;
  p.admaMiddleStep = p.admaMiddleTileW - dep;

  p.admaTopRowDistance = inputWidth * (p.admaTopTileH - dep) - inputWidth + p.admaRightTileW;
  p.admaMidRowDistance = inputWidth * (p.admaMiddleTileH - dep) - inputWidth + p.admaRightTileW;

  p.admaInputSpace = inputHeight * inputWidth;

  p.admaTopBottomLeftPad = (p.admaLeftTileW + pad) * pad;
  p.admaTopBottomMiddlePad = p.admaMiddleTileW * pad;
  p.admaTopBottomRightPad = (p.admaRightTileW + pad) * pad;
  p.admaSidePad = (wCount == 1) ? (2 * pad) : pad;

  // WDMA Parameters
  p.wdmaStartAddress = kernelAddress;
  p.wdmaOutputHCount = hCount;
  p.wdmaOutputWCount = wCount;
  p.wdmaKernelBlockCount = divRoundUp(outputChannels, b) * divRoundUp(inputChannels, b) * kernelHeight * kernelWidth;

  // FDMA Parameters
  //bool enableBnq = true;
  const uint32_t dataWidth = (enable_bnq) ? b * 2 : b * 16;
  const uint32_t avalonDataWidth = (enable_bnq) ? b * 2 : 128;
  auto bytesPerElement = dataWidth / 8;
  auto wordsPerElement = dataWidth / avalonDataWidth;

  auto fdmaRowToRowDistance = [&](uint32_t outputWidth, uint32_t tileWidth) {
    return (outputWidth - tileWidth) * wordsPerElement + 1;
  };

  p.fdmaOutputAddress = outputAddress;

  p.fdmaOutputHCount = hCount;
  p.fdmaOutputWCount = wCount;
  p.fdmaOutputCCount = divRoundUp(outputChannels, b);

  p.fdmaRegularTileH = outputTileHeight;
  p.fdmaLastTileH = outputHeight - (hCount - 1)  * outputTileHeight;

  p.fdmaRegularTileW = outputTileWidth;
  p.fdmaLastTileW = outputWidth - (wCount - 1)  * outputTileWidth;

  p.fdmaRegularRowToRowDistance = fdmaRowToRowDistance(outputWidth, p.fdmaRegularTileW);
  p.fdmaLastRowToRowDistance = fdmaRowToRowDistance(outputWidth, p.fdmaLastTileW);

  p.fdmaOutputSpace = outputHeight * outputWidth;
  p.fdmaRowDistance = outputWidth * p.fdmaRegularTileH - outputWidth + p.fdmaLastTileW;

  // A2F Parameters
  p.a2fInputCCount = p.admaInputCCount;
  p.a2fKernelVCount = kernelHeight;
  p.a2fKernelHCount = kernelWidth;

  if (kernelHeight == 1) {
    p.a2fTileStep = 1u;
    p.a2fTileGap = 1u;
  }
  else {
    // TODO: 3x3 stride one assumed here
    p.a2fTileStep = 1u;
    p.a2fTileGap = 3u;
  }

  p.a2fOutputHCount = hCount;
  p.a2fOutputWCount = wCount;

  p.a2fRegularTileH = outputTileHeight;
  p.a2fLastTileH = outputHeight - (hCount - 1) * p.a2fRegularTileH;
  p.a2fRegularTileW = outputTileWidth;
  p.a2fLastTileW = outputWidth - (wCount - 1) * p.a2fRegularTileW;

  p.qdmaStartAddress = thresholdAddress;
  p.bnqEnable = enable_bnq ? 1 : 0;

  return p;
}

void RunTCA(unsigned long input_addr, unsigned long output_addr, unsigned long kernel_addr,
  unsigned long thresholds_addr, unsigned in_w, unsigned in_h, unsigned in_c, unsigned nbits_in_data,
  unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w, unsigned k_h, unsigned pad, unsigned stride) {

  unsigned use_threshold = (thresholds_addr != 0) ? 1 : 0;

  static SimpleMappedMem csr_mmap(HPS_TO_FPGA_LW_BASE, 0xFF);
  static volatile uint32_t* csr = reinterpret_cast<uint32_t*>(csr_mmap.get());
    auto tileWidth = 32u;
    auto tileHeight = 32u;
    auto p = calcParameters(in_h, in_w, in_c, tileWidth, tileHeight, out_c, k_h, k_w, input_addr, kernel_addr, thresholds_addr, output_addr, use_threshold == 1);

    csr[Csr::admaInputAddress] = p.admaInputAddress;
    csr[Csr::admaInputHCount] = p.admaInputHCount;
    csr[Csr::admaInputWCount] = p.admaInputWCount;
    csr[Csr::admaInputCCount] = p.admaInputCCount;
    csr[Csr::admaTopTileH] = p.admaTopTileH;
    csr[Csr::admaMiddleTileH] = p.admaMiddleTileH;
    csr[Csr::admaBottomTileH] = p.admaBottomTileH;
    csr[Csr::admaLeftTileW] = p.admaLeftTileW;
    csr[Csr::admaMiddleTileW] = p.admaMiddleTileW;
    csr[Csr::admaRightTileW] = p.admaRightTileW;
    csr[Csr::admaLeftRowToRowDistance] = p.admaLeftRowToRowDistance;
    csr[Csr::admaMiddleRowToRowDistance] = p.admaMiddleRowToRowDistance;
    csr[Csr::admaRightRowToRowDistance] = p.admaRightRowToRowDistance;
    csr[Csr::admaLeftStep] = p.admaLeftStep;
    csr[Csr::admaMiddleStep] = p.admaMiddleStep;
    csr[Csr::admaTopRowDistance] = p.admaTopRowDistance;
    csr[Csr::admaMidRowDistance] = p.admaMidRowDistance;
    csr[Csr::admaInputSpace] = p.admaInputSpace;
    csr[Csr::admaTopBottomLeftPad] = p.admaTopBottomLeftPad;
    csr[Csr::admaTopBottomMiddlePad] = p.admaTopBottomMiddlePad;
    csr[Csr::admaTopBottomRightPad] = p.admaTopBottomRightPad;
    csr[Csr::admaSidePad] = p.admaSidePad;
    csr[Csr::wdmaStartAddress] = p.wdmaStartAddress;
    csr[Csr::wdmaOutputHCount] = p.wdmaOutputHCount;
    csr[Csr::wdmaOutputWCount] = p.wdmaOutputWCount;
    csr[Csr::wdmaKernelBlockCount] = p.wdmaKernelBlockCount;
    csr[Csr::fdmaOutputAddress] = p.fdmaOutputAddress;
    csr[Csr::fdmaOutputHCount] = p.fdmaOutputHCount;
    csr[Csr::fdmaOutputWCount] = p.fdmaOutputWCount;
    csr[Csr::fdmaOutputCCount] = p.fdmaOutputCCount;
    csr[Csr::fdmaRegularTileH] = p.fdmaRegularTileH;
    csr[Csr::fdmaLastTileH] = p.fdmaLastTileH;
    csr[Csr::fdmaRegularTileW] = p.fdmaRegularTileW;
    csr[Csr::fdmaLastTileW] = p.fdmaLastTileW;
    csr[Csr::fdmaRegularRowToRowDistance] = p.fdmaRegularRowToRowDistance;
    csr[Csr::fdmaLastRowToRowDistance] = p.fdmaLastRowToRowDistance;
    csr[Csr::fdmaOutputSpace] = p.fdmaOutputSpace;
    csr[Csr::fdmaRowDistance] = p.fdmaRowDistance;
    csr[Csr::a2fInputCCount] = p.a2fInputCCount;
    csr[Csr::a2fKernelVCount] = p.a2fKernelVCount;
    csr[Csr::a2fKernelHCount] = p.a2fKernelHCount;
    csr[Csr::a2fTileStep] = p.a2fTileStep;
    csr[Csr::a2fTileGap] = p.a2fTileGap;
    csr[Csr::a2fOutputHCount] = p.a2fOutputHCount;
    csr[Csr::a2fOutputWCount] = p.a2fOutputWCount;
    csr[Csr::a2fRegularTileH] = p.a2fRegularTileH;
    csr[Csr::a2fLastTileH] = p.a2fLastTileH;
    csr[Csr::a2fRegularTileW] = p.a2fRegularTileW;
    csr[Csr::a2fLastTileW] = p.a2fLastTileW;
    csr[Csr::qdmaStartAddress] = p.qdmaStartAddress;
    csr[Csr::bnqEnable] = p.bnqEnable;

    // std::cout << "Status " << csr[Csr::statusRegister] << std::endl;
    csr[Csr::start] = 1;

    // std::cout << "Status " << csr[Csr::statusRegister] << std::endl;
    while (csr[Csr::statusRegister] != 127) {
        // std::cout << "Status " << csr[Csr::statusRegister] << std::endl;
        continue;
    }
}

} // namespace de10_nano
