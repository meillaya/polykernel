//===- Wavelet.cpp - 32-bit dataflow wavelet -------------------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). See Wavelet.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Wavelet.h"

namespace polykernel::dataflow {

Wavelet MakeWavelet(uint16_t data, uint16_t control) {
  Wavelet w;
  w.data = data;
  w.control = control;
  return w;
}

} // namespace polykernel::dataflow
