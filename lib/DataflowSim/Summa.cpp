//===- Summa.cpp - SUMMA matmul mapping on the dataflow fabric --*- C++ -*-===//
//
// PolyKernel SUMMA matmul mapping + lowering (Todo 37 / Wave 7). See Summa.h.
//
// The simulator FUNCTIONALLY EXECUTES the SUMMA tile program: it broadcasts the
// A panels EAST and the B panels SOUTH as wavelets across the Grid (Todo 35) via
// dataflow receive-and-retransmit tasks (each PE receives a wavelet into its
// local buffer and retransmits it along the row/column - a fabric broadcast),
// runs the per-PE dataflow rendezvous (Todo 36 Scheduler: the compute task fires
// only after BOTH panels arrive), and accumulates the output-stationary C tiles
// with the CE FMAC (Todo 35 Pe::Compute().Fma). The numerical output is
// validated vs a golden matmul in dataflow_correctness_test (contract C).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Summa.h"

#include "PolyKernel/Dataflow/Scheduler.h"
#include "PolyKernel/Dataflow/Task.h"
#include "PolyKernel/Dataflow/Wavelet.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace polykernel::dataflow {

namespace {

// IEEE-754 half (1-5-10) <-> float, mirroring Pe.cpp's CE datapath conversion.
// The SUMMA inputs are small integers (exactly representable), and every partial
// product/sum stays within fp16's exact-integer range, so the FMAC accumulate is
// bit-exact vs the fp32 golden reference.
float HalfToFloat(uint16_t h) {
  const uint32_t sign = (uint32_t{h} & 0x8000u) << 16;
  uint32_t exp = (uint32_t{h} >> 10) & 0x1Fu;
  uint32_t mant = uint32_t{h} & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(f);
}

uint16_t FloatToHalf(float value) {
  const uint32_t f = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (f >> 16) & 0x8000u;
  const uint32_t fexp = (f >> 23) & 0xFFu;
  uint32_t mant = f & 0x7FFFFFu;
  if (fexp == 0xFFu)
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  const int32_t exp = static_cast<int32_t>(fexp) - 127 + 15;
  if (exp >= 0x1F)
    return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    return static_cast<uint16_t>(sign | (mant >> shift));
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

// A pending receive-and-retransmit: a wavelet a PE consumed and must retransmit
// along the broadcast direction (held until the fabric VC has room -> lossless).
struct Retx {
  Color color;
  Dir dir;
  Wavelet w;
};

// Per-PE SUMMA state. cTile is the output-stationary C tile resident in the PE's
// local SRAM (it never moves); aBuf/bBuf collect the broadcast A/B panels; outbox
// holds wavelets the PE consumed and must retransmit to continue the broadcast.
struct PeState {
  std::vector<float> aBuf;  // T*tileK A-panel elements (indexed by wavelet ctrl).
  std::vector<float> bBuf;  // tileK*T B-panel elements.
  std::vector<float> cTile; // T*T output-stationary C tile (local SRAM).
  std::vector<Retx> outbox; // pending retransmits (consume-and-retransmit).
  int aCount = 0;           // A-panel elements arrived this step.
  int bCount = 0;           // B-panel elements arrived this step.
  int stepsDone = 0;        // SUMMA steps completed (panel accumulates fired).
  bool aToken = false;      // x_done (@activate) token delivered.
  bool bToken = false;      // y_done (@unblock) token delivered.
};

// The CE FMAC outer-product accumulate: C_tile += A_panel * B_panel. The inner
// reduction over the K-slab runs through the 4-lane FMAC (lanes vectorise across
// four output columns), modelling the @fmacs outer product. `ce` is the PE's own
// CE (Todo 35 Pe::Compute()).
void PanelAccumulate(PeState &s, const Ce &ce, int T, int TK) {
  for (int i = 0; i < T; ++i) {
    for (int jb = 0; jb < T; jb += Ce::kLanes) {
      Ce::LaneVec acc;
      for (int l = 0; l < Ce::kLanes; ++l) {
        const int j = jb + l;
        acc[l] = (j < T) ? FloatToHalf(s.cTile[static_cast<std::size_t>(i) * T + j])
                         : uint16_t{0};
      }
      for (int kk = 0; kk < TK; ++kk) {
        Ce::LaneVec av, bv;
        const uint16_t ah =
            FloatToHalf(s.aBuf[static_cast<std::size_t>(i) * TK + kk]);
        for (int l = 0; l < Ce::kLanes; ++l)
          av[l] = ah;
        for (int l = 0; l < Ce::kLanes; ++l) {
          const int j = jb + l;
          bv[l] = (j < T)
                      ? FloatToHalf(s.bBuf[static_cast<std::size_t>(kk) * T + j])
                      : uint16_t{0};
        }
        acc = ce.Fma(av, bv, acc); // FMAC: acc += av * bv (per lane).
      }
      for (int l = 0; l < Ce::kLanes; ++l) {
        const int j = jb + l;
        if (j < T)
          s.cTile[static_cast<std::size_t>(i) * T + j] = HalfToFloat(acc[l]);
      }
    }
  }
}

} // namespace

void ConfigureSummaRoutes(Grid &grid, const SummaPlan &plan) {
  const int P = plan.gridP;
  assert(grid.Width() == P && grid.Height() == P &&
         "ConfigureSummaRoutes: grid must be gridP x gridP");
  for (int y = 0; y < P; ++y) {
    for (int x = 0; x < P; ++x) {
      // A broadcast EAST along the row: receive from the West neighbour, re-emit
      // East (| Ramp for a local CE tap). The east edge taps only (no off-grid
      // forwarding -> no drops).
      const DirMask aTx = (x < P - 1) ? (Dir::East | Dir::Ramp) : DirMask(Dir::Ramp);
      for (const Color c : {plan.aColor[0], plan.aColor[1]})
        grid.SetRoute(x, y, c, Dir::West, aTx);
      // B broadcast SOUTH along the column: receive from the North neighbour,
      // re-emit South (| Ramp). The south edge (y == 0) taps only.
      const DirMask bTx = (y > 0) ? (Dir::South | Dir::Ramp) : DirMask(Dir::Ramp);
      for (const Color c : {plan.bColor[0], plan.bColor[1]})
        grid.SetRoute(x, y, c, Dir::North, bTx);
    }
  }
}

SummaResult RunSumma(const SummaPlan &plan, const std::vector<float> &a,
                     const std::vector<float> &b, bool breakRendezvous) {
  const int P = plan.gridP;
  const int T = plan.tileM;
  const int TK = plan.tileK;
  const int panelSize = T * TK; // A-panel elements == B-panel elements (square).
  assert(plan.tileN == T && "SUMMA tiles are square (tileM == tileN)");
  assert(plan.m == P * T && plan.n == P * T && plan.k == plan.steps * TK);
  assert(static_cast<int>(a.size()) == plan.m * plan.k);
  assert(static_cast<int>(b.size()) == plan.k * plan.n);

  Grid grid(P, P);
  // The broadcast is realised by dataflow receive-and-retransmit tasks (below),
  // so the data colors are NOT auto-forwarded by the router (that would race the
  // tasks). The A-east / B-south route pattern is expressed by ConfigureSummaRoutes
  // (exercised in summa_test); here the tasks move wavelets east/south by hopping
  // them one link per cycle via Grid::Inject + Grid::Step.

  std::vector<PeState> st(static_cast<std::size_t>(P) * P);
  for (PeState &s : st) {
    s.aBuf.assign(panelSize, 0.f);
    s.bBuf.assign(panelSize, 0.f);
    s.cTile.assign(static_cast<std::size_t>(T) * T, 0.f);
  }

  SummaResult result;
  uint64_t panelComputes = 0;

  // One scheduler per PE, observing that PE's router. Tasks (registered order =
  // pick priority): four single-input DRAIN tasks (A0/A1/B0/B1) that receive the
  // broadcast wavelets into aBuf/bBuf and retransmit them along the row/column
  // (the fabric broadcast), plus the COMPUTE task that performs the panel
  // accumulate on the x_done/y_done rendezvous.
  std::vector<std::unique_ptr<Scheduler>> scheds;
  scheds.reserve(st.size());
  std::vector<int> computeIds(st.size());
  for (int idx = 0; idx < static_cast<int>(st.size()); ++idx) {
    const int x = idx % P;
    const int y = idx / P;
    PeState &pst = st[idx];
    auto sched = std::make_unique<Scheduler>(grid.At(x, y).GetRouter());

    // Drain A (colors A0, A1): receive into aBuf, retransmit EAST (x < P-1).
    for (const Color c : {plan.aColor[0], plan.aColor[1]}) {
      Task t(static_cast<int>(sched->NumTasks()), TaskKind::Data);
      t.AddInput(c);
      t.SetHandler([&pst, x, c, P](Task &tk) {
        const Wavelet w = tk.LastInputs()[0];
        pst.aBuf[w.control] = HalfToFloat(w.data);
        ++pst.aCount;
        if (x < P - 1)
          pst.outbox.push_back(Retx{c, Dir::East, w}); // broadcast onward east.
      });
      const int id = sched->AddTask(std::move(t));
      sched->Activate(id);
    }
    // Drain B (colors B0, B1): receive into bBuf, retransmit SOUTH (y > 0).
    for (const Color c : {plan.bColor[0], plan.bColor[1]}) {
      Task t(static_cast<int>(sched->NumTasks()), TaskKind::Data);
      t.AddInput(c);
      t.SetHandler([&pst, y, c](Task &tk) {
        const Wavelet w = tk.LastInputs()[0];
        pst.bBuf[w.control] = HalfToFloat(w.data);
        ++pst.bCount;
        if (y > 0)
          pst.outbox.push_back(Retx{c, Dir::South, w}); // broadcast onward south.
      });
      const int id = sched->AddTask(std::move(t));
      sched->Activate(id);
    }
    // Compute task: the panel accumulate, fired by the x_done/y_done rendezvous.
    // Normal: a DATA task bound to BOTH rendezvous colors (xDone, yDone) that
    //   fires only after BOTH tokens arrive; it starts deactivated so x_done ->
    //   @activate and y_done -> @unblock gate the fire (the data-task input
    //   condition additionally requires both tokens present).
    // Broken: bound to ONLY xDone (fires on the A panel alone -> wrong output).
    {
      Task t(static_cast<int>(sched->NumTasks()), TaskKind::Data);
      t.AddInput(plan.xDone);
      if (!breakRendezvous)
        t.AddInput(plan.yDone);
      t.SetHandler([&, x, y, idx](Task &) {
        PanelAccumulate(st[idx], grid.At(x, y).Compute(), T, TK);
        // Reset for the next SUMMA step (ping-pong reuses the buffers).
        std::fill(st[idx].aBuf.begin(), st[idx].aBuf.end(), 0.f);
        std::fill(st[idx].bBuf.begin(), st[idx].bBuf.end(), 0.f);
        st[idx].aCount = 0;
        st[idx].bCount = 0;
        st[idx].aToken = false;
        st[idx].bToken = false;
        ++st[idx].stepsDone;
        ++panelComputes;
      });
      const int id = sched->AddTask(std::move(t));
      computeIds[idx] = id;
      if (breakRendezvous)
        sched->Activate(id); // broken: live from the start, fires on A alone.
      // Normal: left deactivated; x_done activates, y_done unblocks.
    }
    scheds.push_back(std::move(sched));
  }

  // A source (edge) injection that seeds a broadcast.
  struct Inj {
    int x, y;
    Color c;
    Wavelet w;
    bool isA;
  };

  const uint64_t kCycleCap = 20000000; // generous guard against non-termination.
  for (int step = 0; step < plan.steps; ++step) {
    const int p = step & 1; // ping-pong color index.
    const Color aCol = plan.aColor[p];
    const Color bCol = plan.bColor[p];

    // Build this step's SOURCE injections (the broadcast seeds at the west edge
    // for A and the north edge for B), interleaved by panel index t. Each seed is
    // injected to the source PE's RAMP (its own local tap); the drain tasks then
    // retransmit it east/south down the row/column.
    std::vector<Inj> pending;
    pending.reserve(static_cast<std::size_t>(2) * P * panelSize);
    for (int t = 0; t < panelSize; ++t) {
      const int i = t / TK, kk = t % TK; // A panel index (row i, slab col kk).
      for (int y = 0; y < P; ++y) {
        const int grow = y * T + i;
        const int gcol = step * TK + kk;
        const float v = a[static_cast<std::size_t>(grow) * plan.k + gcol];
        pending.push_back({0, y, aCol,
                           MakeWavelet(FloatToHalf(v), static_cast<uint16_t>(t)),
                           true});
      }
      if (!breakRendezvous) {
        const int bkk = t / T, j = t % T; // B panel index (slab row kk, col j).
        for (int x = 0; x < P; ++x) {
          const int grow = step * TK + bkk;
          const int gcol = x * T + j;
          const float v = b[static_cast<std::size_t>(grow) * plan.n + gcol];
          pending.push_back({x, P - 1, bCol,
                             MakeWavelet(FloatToHalf(v), static_cast<uint16_t>(t)),
                             false});
        }
      }
    }

    std::size_t cursor = 0;
    while (true) {
      // Flush pending retransmits (lossless: leave a wavelet buffered if its VC
      // is full, retry next cycle).
      bool outboxPending = false;
      for (int idx = 0; idx < static_cast<int>(st.size()); ++idx) {
        PeState &pst = st[idx];
        const int x = idx % P, y = idx / P;
        std::size_t k = 0;
        while (k < pst.outbox.size()) {
          const Retx &r = pst.outbox[k];
          if (grid.Inject(x, y, r.color, r.dir, r.w))
            ++k;
          else
            break; // VC full: retry next cycle.
        }
        if (k > 0)
          pst.outbox.erase(pst.outbox.begin(), pst.outbox.begin() + k);
        if (!pst.outbox.empty())
          outboxPending = true;
      }
      // Inject source wavelets (seeds) as the source VCs accept them.
      while (cursor < pending.size()) {
        const Inj &ij = pending[cursor];
        if (grid.Inject(ij.x, ij.y, ij.c, Dir::Ramp, ij.w)) {
          if (ij.isA)
            ++result.aWavelets;
          else
            ++result.bWavelets;
          ++cursor;
        } else {
          break; // source VC full: step the fabric to make room, then retry.
        }
      }
      grid.Step(); // single-cycle hop: seeds/retransmits advance one link.
      for (auto &s : scheds)
        s->RunToQuiescence(); // receive arrived wavelets, queue retransmits.

      // x_done / y_done: when a panel is complete, deliver its rendezvous token
      // and gate the compute task (x_done -> @activate, y_done -> @unblock).
      for (int idx = 0; idx < static_cast<int>(st.size()); ++idx) {
        PeState &pst = st[idx];
        const int x = idx % P, y = idx / P;
        if (!pst.aToken && pst.aCount >= panelSize) {
          scheds[idx]->Activate(computeIds[idx]); // x_done -> @activate(compute).
          grid.At(x, y).GetRouter().Rx(Dir::Ramp, plan.xDone, MakeWavelet(1, 0));
          pst.aToken = true;
        }
        if (!breakRendezvous && !pst.bToken && pst.bCount >= panelSize) {
          scheds[idx]->Unblock(computeIds[idx]); // y_done -> @unblock(compute).
          grid.At(x, y).GetRouter().Rx(Dir::Ramp, plan.yDone, MakeWavelet(1, 0));
          pst.bToken = true;
        }
      }
      for (auto &s : scheds)
        s->RunToQuiescence(); // fire the compute where both tokens arrived.

      bool allDone = true;
      for (const PeState &pst : st)
        if (pst.stepsDone < step + 1)
          allDone = false;
      if (cursor == pending.size() && !outboxPending && allDone) {
        bool fabricEmpty = true;
        for (int y = 0; y < P; ++y)
          for (int x = 0; x < P; ++x)
            if (grid.At(x, y).GetRouter().Buffered() > 0)
              fabricEmpty = false;
        if (fabricEmpty)
          break;
      }
      assert(grid.Cycle() < kCycleCap && "RunSumma: fabric did not quiesce");
      if (grid.Cycle() >= kCycleCap)
        break;
    }
  }

  // Gather the output-stationary C tiles into the full m x n result.
  result.c.assign(static_cast<std::size_t>(plan.m) * plan.n, 0.f);
  for (int y = 0; y < P; ++y)
    for (int x = 0; x < P; ++x) {
      const PeState &pst = st[static_cast<std::size_t>(y) * P + x];
      for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j) {
          const int grow = y * T + i;
          const int gcol = x * T + j;
          result.c[static_cast<std::size_t>(grow) * plan.n + gcol] =
              pst.cTile[static_cast<std::size_t>(i) * T + j];
        }
    }

  result.cycles = grid.Cycle();
  result.panelComputes = panelComputes;
  result.droppedOffGrid = grid.DroppedOffGrid();
  return result;
}

} // namespace polykernel::dataflow
