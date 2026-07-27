//===- scheduler_test.cpp - Dataflow task model + scheduler ---*- C++ -*-===//
//
// PolyKernel dataflow task model + scheduler (Todo 36 / Wave 7). Suite
// `scheduler` so `ctest -R scheduler` discovers these.
//
// Asserts the dataflow task model + rendezvous:
//   - the THREE task types (data = wavelet-triggered/bound to a color, local =
//     self-triggered via @activate, control),
//   - a COMPUTE task (a data task with two bound input colors) fires only after
//     BOTH input wavelets arrive (the dataflow rendezvous) - it does NOT fire
//     after only one,
//   - a BLOCKED task does NOT fire until @unblock (no premature fire),
//   - a data task fires on wavelet arrival on its bound color (including arrival
//     driven through the Todo 35 Grid fabric),
//   - a local task fires on @activate,
//   - the TaskPicker selects a task only when activated AND unblocked AND (data)
//     its inputs have arrived.
//
// Terminology: there are NO @compute/@data decorators - `compute` is an exported
// host-callable convention (the task handler). This is a GPU-free functional
// SIMULATOR, not real CSL and not Cerebras hardware.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Grid.h"
#include "PolyKernel/Dataflow/Scheduler.h"

#include "gtest/gtest.h"

namespace {

using polykernel::dataflow::Color;
using polykernel::dataflow::Dir;
using polykernel::dataflow::Grid;
using polykernel::dataflow::MakeWavelet;
using polykernel::dataflow::Router;
using polykernel::dataflow::Scheduler;
using polykernel::dataflow::Task;
using polykernel::dataflow::TaskKind;
using polykernel::dataflow::TaskKindName;
using polykernel::dataflow::TaskPicker;
using polykernel::dataflow::Wavelet;

// The three dataflow task types exist and are named with CSL terminology.
TEST(scheduler, ThreeTaskTypes) {
  EXPECT_STREQ(TaskKindName(TaskKind::Data), "data");
  EXPECT_STREQ(TaskKindName(TaskKind::Local), "local");
  EXPECT_STREQ(TaskKindName(TaskKind::Control), "control");

  Task data(0, TaskKind::Data);
  Task local(1, TaskKind::Local);
  Task control(2, TaskKind::Control);
  EXPECT_EQ(data.Kind(), TaskKind::Data);
  EXPECT_EQ(local.Kind(), TaskKind::Local);
  EXPECT_EQ(control.Kind(), TaskKind::Control);
}

// KEY rendezvous: a compute task (a data task bound to TWO input colors) fires
// only after BOTH input wavelets arrive - it does NOT fire after only one.
TEST(scheduler, ComputeTaskFiresOnlyAfterBothInputsArrive) {
  Router r;
  const Color a = *Color::Routable(0);
  const Color b = *Color::Routable(1);

  Scheduler sched(r);
  Task compute(0, TaskKind::Data); // the exported host-callable `compute`.
  compute.AddInput(a);             // two bound input colors => a compute task.
  compute.AddInput(b);
  int runs = 0;
  int sum = 0;
  compute.SetHandler([&](Task &t) {
    ++runs;
    for (const Wavelet &w : t.LastInputs())
      sum += w.data;
  });
  const int id = sched.AddTask(std::move(compute));
  sched.Activate(id); // activated + (default) unblocked.

  // No inputs yet: not runnable.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 0u);

  // Only input `a` arrives: the rendezvous is INCOMPLETE -> no fire.
  const Wavelet wa = MakeWavelet(10, 0);
  ASSERT_TRUE(r.Rx(Dir::West, a, wa));
  EXPECT_EQ(sched.Pick(), nullptr); // picker rejects: one input missing.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 0u); // did NOT fire after only one.
  EXPECT_EQ(runs, 0);

  // Input `b` arrives: BOTH inputs present -> the compute task fires.
  const Wavelet wb = MakeWavelet(20, 0);
  ASSERT_TRUE(r.Rx(Dir::West, b, wb));
  EXPECT_EQ(sched.Step(), id);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
  EXPECT_EQ(runs, 1);
  EXPECT_EQ(sum, 30); // compute saw BOTH operands (10 + 20).
  // The fire consumed both input wavelets, in AddInput order.
  ASSERT_EQ(sched.GetTask(id).LastInputs().size(), 2u);
  EXPECT_EQ(sched.GetTask(id).LastInputs()[0], wa);
  EXPECT_EQ(sched.GetTask(id).LastInputs()[1], wb);
  EXPECT_EQ(r.InputSize(a), 0u);
  EXPECT_EQ(r.InputSize(b), 0u);

  // Inputs consumed: nothing runnable again until more wavelets arrive.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
}

// KEY negative: a task that is activated but BLOCKED does NOT fire (no premature
// fire) until @unblock releases it - even with its input wavelet present.
TEST(scheduler, BlockedTaskDoesNotFireUntilUnblocked) {
  Router r;
  const Color c = *Color::Routable(2);

  Scheduler sched(r);
  Task t(0, TaskKind::Data);
  t.AddInput(c);
  const int id = sched.AddTask(std::move(t));
  sched.Activate(id); // activated...
  sched.Block(id);    // ...but BLOCKED.

  // Its input wavelet arrives, yet it must NOT fire while blocked.
  ASSERT_TRUE(r.Rx(Dir::West, c, MakeWavelet(0x0042, 0)));
  EXPECT_EQ(sched.Pick(), nullptr); // blocked => not runnable.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 0u); // no premature fire.
  EXPECT_EQ(r.InputSize(c), 1u);                // wavelet still buffered.

  // @unblock releases the gate: now it fires.
  sched.Unblock(id);
  EXPECT_EQ(sched.Step(), id);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
  EXPECT_EQ(r.InputSize(c), 0u); // consumed on fire.
}

// A data task fires on wavelet arrival on its bound color (and not before).
TEST(scheduler, DataTaskFiresOnWaveletArrival) {
  Router r;
  const Color c = *Color::Routable(3);

  Scheduler sched(r);
  Task t(0, TaskKind::Data);
  t.AddInput(c); // bound to color c (its data_task_id).
  const int id = sched.AddTask(std::move(t));
  sched.Activate(id);

  // No wavelet on the bound color yet: no fire.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 0u);

  // A wavelet arrives on the bound color: the data task fires.
  const Wavelet w = MakeWavelet(0x00FF, 0x00A5);
  ASSERT_TRUE(r.Rx(Dir::North, c, w));
  EXPECT_EQ(sched.Step(), id);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
  ASSERT_EQ(sched.GetTask(id).LastInputs().size(), 1u);
  EXPECT_EQ(sched.GetTask(id).LastInputs()[0], w);

  // The wavelet was consumed: no re-fire without a fresh arrival.
  EXPECT_EQ(sched.Step(), -1);
}

// A data task fires on wavelet arrival driven THROUGH the Todo 35 Grid fabric
// (single-cycle hop delivers the wavelet to the task's PE router input VC).
TEST(scheduler, DataTaskFiresOnWaveletArrivalViaGrid) {
  Grid g;
  const Color c = *Color::Routable(5);
  Router &r = g.At(1, 0).GetRouter(); // the task's PE router.

  Scheduler sched(r);
  Task t(0, TaskKind::Data);
  t.AddInput(c);
  const int id = sched.AddTask(std::move(t));
  sched.Activate(id);

  // No wavelet yet: not runnable.
  EXPECT_EQ(sched.Step(), -1);

  // Inject one hop west of the PE and step the fabric: the wavelet arrives on c.
  const Wavelet w = MakeWavelet(0x0042, 0x0);
  ASSERT_TRUE(g.Inject(0, 0, c, Dir::East, w));
  g.Step(); // single-cycle hop -> PE (1,0) input VC.
  ASSERT_EQ(r.InputSize(c), 1u); // wavelet arrived on the bound color.

  EXPECT_EQ(sched.Step(), id); // data task fires on arrival.
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
  ASSERT_EQ(sched.GetTask(id).LastInputs().size(), 1u);
  EXPECT_EQ(sched.GetTask(id).LastInputs()[0], w);
}

// A local task is self-triggered via @activate: it does not fire until activated
// (and waits on no input wavelets).
TEST(scheduler, LocalTaskFiresOnActivate) {
  Router r;
  Scheduler sched(r);
  const int id = sched.AddTask(Task(0, TaskKind::Local));

  // Not activated yet: no fire.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 0u);

  // @activate is the local task's trigger: it fires (no input wavelets needed).
  sched.Activate(id);
  EXPECT_EQ(sched.Step(), id);
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
  EXPECT_TRUE(sched.GetTask(id).LastInputs().empty()); // no operands.
}

// A control task is gated by the same rendezvous: blocked => no fire; activated
// + unblocked => fire.
TEST(scheduler, ControlTaskGatedByRendezvous) {
  Router r;
  Scheduler sched(r);
  const int id = sched.AddTask(Task(0, TaskKind::Control));
  sched.Activate(id);
  sched.Block(id);
  EXPECT_EQ(sched.Step(), -1); // blocked: no fire.
  sched.Unblock(id);
  EXPECT_EQ(sched.Step(), id); // released: fires.
  EXPECT_EQ(sched.GetTask(id).FireCount(), 1u);
}

// The TaskPicker selection matrix: a task is runnable iff activated AND
// unblocked AND (for data tasks) every bound input color has a wavelet.
TEST(scheduler, TaskPickerSelectsActivatedAndUnblocked) {
  Router r;
  const Color c = *Color::Routable(4);

  // --- data task: needs activate + unblock + wavelet arrival. ---
  Task data(0, TaskKind::Data);
  data.AddInput(c);
  EXPECT_FALSE(TaskPicker::IsRunnable(data, r)); // not activated.
  data.Activate();
  EXPECT_FALSE(TaskPicker::IsRunnable(data, r)); // activated, but no wavelet.
  ASSERT_TRUE(r.Rx(Dir::West, c, MakeWavelet(1, 0)));
  EXPECT_TRUE(TaskPicker::IsRunnable(data, r)); // activated + wavelet arrived.
  data.Block();
  EXPECT_FALSE(TaskPicker::IsRunnable(data, r)); // blocked gates it off.
  data.Unblock();
  EXPECT_TRUE(TaskPicker::IsRunnable(data, r)); // unblocked: runnable again.

  // --- local task: needs activate + unblock only (no input wavelets). ---
  Task local(1, TaskKind::Local);
  EXPECT_FALSE(TaskPicker::IsRunnable(local, r)); // not activated.
  local.Activate();
  EXPECT_TRUE(TaskPicker::IsRunnable(local, r)); // activated: runnable.
  local.Block();
  EXPECT_FALSE(TaskPicker::IsRunnable(local, r)); // blocked gates it off.
}

// @activate/@block/@unblock drive Task::IsEligible (activated AND unblocked).
TEST(scheduler, ActivateBlockUnblockRendezvous) {
  Task t(0, TaskKind::Local);
  EXPECT_FALSE(t.IsEligible()); // default: not activated.
  t.Activate();
  EXPECT_TRUE(t.IsEligible()); // activated + unblocked.
  t.Block();
  EXPECT_FALSE(t.IsEligible()); // @block gates it off.
  EXPECT_TRUE(t.IsActivated()); // still activated, just blocked.
  t.Unblock();
  EXPECT_TRUE(t.IsEligible()); // @unblock releases it.
}

// The picker selects the lowest-id runnable task first.
TEST(scheduler, PickerSelectsLowestIdFirst) {
  Router r;
  Scheduler sched(r);
  const int hi = sched.AddTask(Task(5, TaskKind::Local));
  const int lo = sched.AddTask(Task(2, TaskKind::Local));
  sched.Activate(hi);
  sched.Activate(lo);
  // Both runnable; the picker walks tasks in registration order, so the first
  // registered (id 5) is picked. Register-order is the documented tie-break.
  Task *picked = sched.Pick();
  ASSERT_NE(picked, nullptr);
  EXPECT_EQ(picked->Id(), hi);
  EXPECT_EQ(lo, 2);
}

} // namespace
