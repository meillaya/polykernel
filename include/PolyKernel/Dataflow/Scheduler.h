//===- Scheduler.h - Dataflow task scheduler (rendezvous) ------*- C++ -*-===//
//
// PolyKernel dataflow task model + scheduler (Todo 36 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware). It builds on the Todo 35 dataflow core: the scheduler observes
// wavelet arrival on a PE's router (Router.h) and fires the data tasks bound to
// the colors those wavelets arrive on.
//
// The Scheduler owns a set of tasks bound to ONE PE router and drives the
// `@activate` / `@block` / `@unblock` rendezvous:
//   - a task runs ONLY when activated AND unblocked,
//   - a data task fires on wavelet arrival on its bound color(s); a compute task
//     (two bound input colors) fires only after BOTH wavelets arrive,
//   - a local task fires on `@activate`,
//   - the TaskPicker selects which runnable task fires next.
//
// Firing a data task consumes one wavelet from each of its bound input colors
// (the task's operands) and invokes its compute handler.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_SCHEDULER_H
#define POLYKERNEL_DATAFLOW_SCHEDULER_H

#include "PolyKernel/Dataflow/Router.h"
#include "PolyKernel/Dataflow/Task.h"
#include "PolyKernel/Dataflow/TaskPicker.h"

#include <cstddef>
#include <deque>

namespace polykernel::dataflow {

class Scheduler {
public:
  // Bind the scheduler to ONE PE router (the fabric the tasks observe / drain).
  explicit Scheduler(Router &router);

  // --- task registration. ---
  // Register a task; returns its id (== task.Id()). The scheduler keeps a stable
  // reference (std::deque), so task pointers/references stay valid across adds.
  int AddTask(Task task);
  std::size_t NumTasks() const { return tasks_.size(); }
  Task &GetTask(int id);
  const Task &GetTask(int id) const;

  // --- @activate / @block / @unblock rendezvous (by task id). ---
  void Activate(int id); // @activate
  void Block(int id);    // @block: gate the task off (no premature fire).
  void Unblock(int id);  // @unblock: release the gate.

  // --- scheduling. ---
  // The task picker: the next runnable task (activated AND unblocked AND, for
  // data tasks, all input wavelets arrived), else nullptr. The returned pointer
  // is stable until the next AddTask.
  Task *Pick();

  // Run one scheduling step: pick a runnable task and fire it (a data task
  // consumes one wavelet per bound input color, then runs its compute handler).
  // Returns the fired task id, or -1 if nothing was runnable.
  int Step();

  // Drain: Step until no task is runnable. Returns the number of tasks fired.
  int RunToQuiescence();

  const Router &GetRouter() const { return *router_; }

private:
  Router *router_;
  std::deque<Task> tasks_;
  TaskPicker picker_;
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_SCHEDULER_H
