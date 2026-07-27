//===- Task.h - Dataflow task model (3 task types) -------------*- C++ -*-===//
//
// PolyKernel dataflow task model + scheduler (Todo 36 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware). It builds on the Todo 35 dataflow core (Color.h / Wavelet.h /
// Router.h): a task is triggered by, and consumes, wavelets that arrive on the
// fabric colors it is bound to.
//
// A *task* is the unit of dataflow-triggered compute. There are THREE task types
// (matching the Cerebras CSL task model):
//   - Data task    - wavelet-triggered, bound to one or more colors (its
//                    data_task_id is the color). It fires only once a wavelet
//                    has arrived on EVERY bound color. A "compute task" is just
//                    a data task bound to two or more input colors: it fires
//                    only after ALL of its input wavelets arrive (rendezvous).
//   - Local task   - self-triggered via `@activate` (no input wavelets).
//   - Control task - a control-flow task, also gated by the rendezvous below.
//
// Rendezvous (`@activate` / `@block` / `@unblock`): a task runs ONLY when it is
// activated AND unblocked. `@block` gates a task off (no premature fire);
// `@unblock` releases it. The TaskPicker (TaskPicker.h) selects a task to run
// exactly when it is activated AND unblocked AND (for data tasks) its input
// wavelets have arrived.
//
// Terminology guardrail: there are NO `@compute` / `@data` decorators in CSL -
// `compute` is an exported host-callable convention (modelled here as the
// task's handler), not a decorator. The data task's trigger is wavelet arrival.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_TASK_H
#define POLYKERNEL_DATAFLOW_TASK_H

#include "PolyKernel/Dataflow/Color.h"
#include "PolyKernel/Dataflow/Wavelet.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace polykernel::dataflow {

//===----------------------------------------------------------------------===//
// TaskKind: the three dataflow task types.
//===----------------------------------------------------------------------===//

enum class TaskKind : uint8_t {
  Data = 0,    // wavelet-triggered, bound to color(s) (the data_task_id).
  Local = 1,   // self-triggered via `@activate` (no input wavelets).
  Control = 2, // control-flow task (gated by the same rendezvous).
};

// Lower-case name of a task kind ("data", "local", "control").
const char *TaskKindName(TaskKind k);

//===----------------------------------------------------------------------===//
// Task: a unit of dataflow-triggered compute.
//===----------------------------------------------------------------------===//

class Task {
public:
  // The exported host-callable `compute` convention (NOT a decorator): invoked
  // when the task fires. Receives the task (so it can read LastInputs()).
  using Handler = std::function<void(Task &)>;

  Task(int id, TaskKind kind);

  int Id() const { return id_; }
  TaskKind Kind() const { return kind_; }

  // --- input colors (the data task's trigger / compute operands). ---
  // Bind an input color. A data task fires only once a wavelet has arrived on
  // EVERY bound color (the dataflow rendezvous); a compute task binds 2+ colors
  // and so fires only after ALL of its inputs arrive. Local/control tasks bind
  // none.
  void AddInput(Color c);
  const std::vector<Color> &Inputs() const { return inputs_; }

  // --- @activate / @block / @unblock rendezvous. ---
  void Activate() { activated_ = true; } // @activate: mark activated.
  void Block() { blocked_ = true; }      // @block: gate the task off.
  void Unblock() { blocked_ = false; }   // @unblock: release the gate.
  bool IsActivated() const { return activated_; }
  bool IsBlocked() const { return blocked_; }

  // The rendezvous eligibility: a task may run only when activated AND
  // unblocked. (For data tasks the TaskPicker additionally requires the input
  // wavelets to have arrived.)
  bool IsEligible() const { return activated_ && !blocked_; }

  // --- compute handler (the exported host-callable convention). ---
  void SetHandler(Handler h) { handler_ = std::move(h); }
  bool HasHandler() const { return static_cast<bool>(handler_); }

  // --- firing. ---
  // Fire the task: record the fire and invoke the compute handler (if any).
  // Called by the Scheduler only once the TaskPicker selects this task.
  void Fire();
  uint64_t FireCount() const { return fire_count_; }

  // The input wavelets consumed on the most recent fire (a data task's
  // operands, in AddInput order). Empty for local/control tasks.
  const std::vector<Wavelet> &LastInputs() const { return last_inputs_; }
  void SetLastInputs(std::vector<Wavelet> w) { last_inputs_ = std::move(w); }

private:
  int id_;
  TaskKind kind_;
  std::vector<Color> inputs_;
  bool activated_ = false;
  bool blocked_ = false;
  uint64_t fire_count_ = 0;
  Handler handler_;
  std::vector<Wavelet> last_inputs_;
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_TASK_H
