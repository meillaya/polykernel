//===- TaskPicker.h - Selects a runnable dataflow task ---------*- C++ -*-===//
//
// PolyKernel dataflow task model + scheduler (Todo 36 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware).
//
// The task picker selects a task to run exactly when it is BOTH activated AND
// unblocked (the `@activate` / `@block` / `@unblock` rendezvous) AND, for a data
// task, a wavelet has arrived on EVERY bound input color (the dataflow trigger /
// rendezvous). A compute task (a data task with two inputs) is therefore picked
// only after BOTH of its input wavelets arrive. Local and control tasks are
// self-triggered by `@activate` and wait on no input wavelets.
//
// The picker is stateless: it observes the tasks and the router's input virtual
// channels (wavelet arrival) and chooses the first runnable task (lowest id).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_TASKPICKER_H
#define POLYKERNEL_DATAFLOW_TASKPICKER_H

#include "PolyKernel/Dataflow/Router.h"
#include "PolyKernel/Dataflow/Task.h"

#include <deque>

namespace polykernel::dataflow {

class TaskPicker {
public:
  // True iff `t` may fire now: activated AND unblocked AND (for data tasks) a
  // wavelet has arrived on EVERY bound input color. A blocked task is NEVER
  // runnable (no premature fire); a data task with a missing input is not
  // runnable until the rendezvous completes.
  static bool IsRunnable(const Task &t, const Router &router);

  // Select the first runnable task (lowest id) from `tasks`, else nullptr.
  static Task *Pick(std::deque<Task> &tasks, const Router &router);
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_TASKPICKER_H
