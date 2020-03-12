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
#include <cstddef>

namespace de10_nano {

//
// TCA
//
enum class Csr {
    start = 0,
    admaInputAddress = 1,
    admaInputHCount = 2,
    admaInputWCount = 3,
    admaInputCCount = 4,
    admaTopTileH = 5,
    admaMiddleTileH = 6,
    admaBottomTileH = 7,
    admaLeftTileW = 8,
    admaMiddleTileW = 9,
    admaRightTileW = 10,
    admaLeftRowToRowDistance = 11,
    admaMiddleRowToRowDistance = 12,
    admaRightRowToRowDistance = 13,
    admaLeftStep = 14,
    admaMiddleStep = 15,
    admaTopRowDistance = 16,
    admaMidRowDistance = 17,
    admaInputSpace = 18,
    admaTopBottomLeftPad = 19,
    admaTopBottomMiddlePad = 20,
    admaTopBottomRightPad = 21,
    admaSidePad = 22,
    wdmaStartAddress = 23,
    wdmaOutputHCount = 24,
    wdmaOutputWCount = 25,
    wdmaKernelBlockCount = 26,
    fdmaOutputAddress = 27,
    fdmaOutputHCount = 28,
    fdmaOutputWCount = 29,
    fdmaOutputCCount = 30,
    fdmaRegularTileH = 31,
    fdmaLastTileH = 32,
    fdmaRegularTileW = 33,
    fdmaLastTileW = 34,
    fdmaRegularRowToRowDistance = 35,
    fdmaLastRowToRowDistance = 36,
    fdmaOutputSpace = 37,
    fdmaRowDistance = 38,
    a2fInputCCount = 39,
    a2fKernelVCount = 40,
    a2fKernelHCount = 41,
    a2fTileStep = 42,
    a2fTileGap = 43,
    a2fOutputHCount = 44,
    a2fOutputWCount = 45,
    a2fRegularTileH = 46,
    a2fLastTileH = 47,
    a2fRegularTileW = 48,
    a2fLastTileW = 49,
    qdmaStartAddress = 50,
    bnqEnable = 51,

    statusRegister = 52,
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

  constexpr std::size_t n_bit = 2;
  constexpr std::size_t maxA = (1 << n_bit) - 1;
  assert(kernelHeight == kernelWidth); // kernel rectangle must be square
  assert(kernelHeight % 2 == 1); // kernel size must be odd
  assert(1 <= kernelHeight && kernelHeight <= 3); // Currently, only 1x1, 3x3 conv are supported
  assert(inputChannels * kernelHeight * kernelWidth * maxA <= std::numeric_limits<BIN_CONV_OUTPUT>::max()); // overflow check

  uint32_t pad = kernelHeight / 2;
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

  p.a2fTileStep = 1u; // stride one assumed
  p.a2fTileGap = kernelHeight; // stride one assumed

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
  unsigned long thresholds_addr, unsigned in_w, unsigned in_h, unsigned in_c,
  unsigned out_w, unsigned out_h, unsigned out_c, unsigned k_w, unsigned k_h, unsigned pad, unsigned stride) {

  unsigned use_threshold = (thresholds_addr != 0) ? 1 : 0;

  static MappedMem csr_mmap(HPS_TO_FPGA_LW_BASE, 0xFF);
  static volatile uint32_t* csr = reinterpret_cast<uint32_t*>(csr_mmap.get());
    auto tileWidth = 32u;
    auto tileHeight = 32u;
    auto p = calcParameters(in_h, in_w, in_c, tileWidth, tileHeight, out_c, k_h, k_w, input_addr, kernel_addr, thresholds_addr, output_addr, use_threshold == 1);

    csr[static_cast<std::size_t>(Csr::admaInputAddress)] = p.admaInputAddress;
    csr[static_cast<std::size_t>(Csr::admaInputHCount)] = p.admaInputHCount;
    csr[static_cast<std::size_t>(Csr::admaInputWCount)] = p.admaInputWCount;
    csr[static_cast<std::size_t>(Csr::admaInputCCount)] = p.admaInputCCount;
    csr[static_cast<std::size_t>(Csr::admaTopTileH)] = p.admaTopTileH;
    csr[static_cast<std::size_t>(Csr::admaMiddleTileH)] = p.admaMiddleTileH;
    csr[static_cast<std::size_t>(Csr::admaBottomTileH)] = p.admaBottomTileH;
    csr[static_cast<std::size_t>(Csr::admaLeftTileW)] = p.admaLeftTileW;
    csr[static_cast<std::size_t>(Csr::admaMiddleTileW)] = p.admaMiddleTileW;
    csr[static_cast<std::size_t>(Csr::admaRightTileW)] = p.admaRightTileW;
    csr[static_cast<std::size_t>(Csr::admaLeftRowToRowDistance)] = p.admaLeftRowToRowDistance;
    csr[static_cast<std::size_t>(Csr::admaMiddleRowToRowDistance)] = p.admaMiddleRowToRowDistance;
    csr[static_cast<std::size_t>(Csr::admaRightRowToRowDistance)] = p.admaRightRowToRowDistance;
    csr[static_cast<std::size_t>(Csr::admaLeftStep)] = p.admaLeftStep;
    csr[static_cast<std::size_t>(Csr::admaMiddleStep)] = p.admaMiddleStep;
    csr[static_cast<std::size_t>(Csr::admaTopRowDistance)] = p.admaTopRowDistance;
    csr[static_cast<std::size_t>(Csr::admaMidRowDistance)] = p.admaMidRowDistance;
    csr[static_cast<std::size_t>(Csr::admaInputSpace)] = p.admaInputSpace;
    csr[static_cast<std::size_t>(Csr::admaTopBottomLeftPad)] = p.admaTopBottomLeftPad;
    csr[static_cast<std::size_t>(Csr::admaTopBottomMiddlePad)] = p.admaTopBottomMiddlePad;
    csr[static_cast<std::size_t>(Csr::admaTopBottomRightPad)] = p.admaTopBottomRightPad;
    csr[static_cast<std::size_t>(Csr::admaSidePad)] = p.admaSidePad;
    csr[static_cast<std::size_t>(Csr::wdmaStartAddress)] = p.wdmaStartAddress;
    csr[static_cast<std::size_t>(Csr::wdmaOutputHCount)] = p.wdmaOutputHCount;
    csr[static_cast<std::size_t>(Csr::wdmaOutputWCount)] = p.wdmaOutputWCount;
    csr[static_cast<std::size_t>(Csr::wdmaKernelBlockCount)] = p.wdmaKernelBlockCount;
    csr[static_cast<std::size_t>(Csr::fdmaOutputAddress)] = p.fdmaOutputAddress;
    csr[static_cast<std::size_t>(Csr::fdmaOutputHCount)] = p.fdmaOutputHCount;
    csr[static_cast<std::size_t>(Csr::fdmaOutputWCount)] = p.fdmaOutputWCount;
    csr[static_cast<std::size_t>(Csr::fdmaOutputCCount)] = p.fdmaOutputCCount;
    csr[static_cast<std::size_t>(Csr::fdmaRegularTileH)] = p.fdmaRegularTileH;
    csr[static_cast<std::size_t>(Csr::fdmaLastTileH)] = p.fdmaLastTileH;
    csr[static_cast<std::size_t>(Csr::fdmaRegularTileW)] = p.fdmaRegularTileW;
    csr[static_cast<std::size_t>(Csr::fdmaLastTileW)] = p.fdmaLastTileW;
    csr[static_cast<std::size_t>(Csr::fdmaRegularRowToRowDistance)] = p.fdmaRegularRowToRowDistance;
    csr[static_cast<std::size_t>(Csr::fdmaLastRowToRowDistance)] = p.fdmaLastRowToRowDistance;
    csr[static_cast<std::size_t>(Csr::fdmaOutputSpace)] = p.fdmaOutputSpace;
    csr[static_cast<std::size_t>(Csr::fdmaRowDistance)] = p.fdmaRowDistance;
    csr[static_cast<std::size_t>(Csr::a2fInputCCount)] = p.a2fInputCCount;
    csr[static_cast<std::size_t>(Csr::a2fKernelVCount)] = p.a2fKernelVCount;
    csr[static_cast<std::size_t>(Csr::a2fKernelHCount)] = p.a2fKernelHCount;
    csr[static_cast<std::size_t>(Csr::a2fTileStep)] = p.a2fTileStep;
    csr[static_cast<std::size_t>(Csr::a2fTileGap)] = p.a2fTileGap;
    csr[static_cast<std::size_t>(Csr::a2fOutputHCount)] = p.a2fOutputHCount;
    csr[static_cast<std::size_t>(Csr::a2fOutputWCount)] = p.a2fOutputWCount;
    csr[static_cast<std::size_t>(Csr::a2fRegularTileH)] = p.a2fRegularTileH;
    csr[static_cast<std::size_t>(Csr::a2fLastTileH)] = p.a2fLastTileH;
    csr[static_cast<std::size_t>(Csr::a2fRegularTileW)] = p.a2fRegularTileW;
    csr[static_cast<std::size_t>(Csr::a2fLastTileW)] = p.a2fLastTileW;
    csr[static_cast<std::size_t>(Csr::qdmaStartAddress)] = p.qdmaStartAddress;
    csr[static_cast<std::size_t>(Csr::bnqEnable)] = p.bnqEnable;

    // std::cout << "Status " << csr[Csr::statusRegister)] << std::endl;
    csr[static_cast<std::size_t>(Csr::start)] = 1;

    // std::cout << "Status " << csr[Csr::statusRegister)] << std::endl;
    while (csr[static_cast<std::size_t>(Csr::statusRegister)] != 127) {
        // std::cout << "Status " << csr[Csr::statusRegister] << std::endl;
        continue;
    }
}

} // namespace de10_nano
