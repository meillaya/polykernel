//===- Scheduler.cpp - Dataflow task scheduler + task picker ---*- C++ -*-===//
//
// PolyKernel dataflow task model + scheduler (Todo 36 / Wave 7). See
// Scheduler.h / TaskPicker.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Scheduler.h"

#include <cassert>
#include <vector>

namespace polykernel::dataflow {

//===----------------------------------------------------------------------===//
// TaskPicker
//===----------------------------------------------------------------------===//

bool TaskPicker::IsRunnable(const Task &t, const Router &router) {
  // The @activate/@block/@unblock rendezvous: a task runs only when activated
  // AND unblocked. A blocked task is NEVER runnable (no premature fire).
  if (t.IsBlocked())
    return false; // @block gate.
  if (!t.IsActivated())
    return false; // @activate gate.
  // A data task is wavelet-triggered: it fires only once a wavelet has arrived
  // on EVERY bound input color. A compute task (two bound inputs) therefore
  // fires only after BOTH wavelets arrive (the dataflow rendezvous).
  if (t.Kind() == TaskKind::Data)
    for (const Color &c : t.Inputs())
      if (router.InputSize(c) == 0)
        return false; // input not yet arrived: rendezvous incomplete.
  // Local + control tasks are self-triggered by @activate (no input wait).
  return true;
}

Task *TaskPicker::Pick(std::deque<Task> &tasks, const Router &router) {
  for (Task &t : tasks)
    if (IsRunnable(t, router))
      return &t;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Scheduler
//===----------------------------------------------------------------------===//

Scheduler::Scheduler(Router &router) : router_(&router) {}

int Scheduler::AddTask(Task task) {
  const int id = task.Id();
  tasks_.push_back(std::move(task));
  return id;
}

Task &Scheduler::GetTask(int id) {
  for (Task &t : tasks_)
    if (t.Id() == id)
      return t;
  assert(false && "Scheduler::GetTask: no such task id");
  return tasks_.front();
}

const Task &Scheduler::GetTask(int id) const {
  for (const Task &t : tasks_)
    if (t.Id() == id)
      return t;
  assert(false && "Scheduler::GetTask: no such task id");
  return tasks_.front();
}

void Scheduler::Activate(int id) { GetTask(id).Activate(); }
void Scheduler::Block(int id) { GetTask(id).Block(); }
void Scheduler::Unblock(int id) { GetTask(id).Unblock(); }

Task *Scheduler::Pick() { return picker_.Pick(tasks_, *router_); }

int Scheduler::Step() {
  Task *t = Pick();
  if (!t)
    return -1; // nothing runnable: no fire this step.
  // A data task consumes one wavelet from each bound input color on fire (its
  // operands, in AddInput order). Local/control tasks consume nothing.
  if (t->Kind() == TaskKind::Data) {
    std::vector<Wavelet> inputs;
    inputs.reserve(t->Inputs().size());
    for (const Color &c : t->Inputs()) {
      auto w = router_->PopInput(c);
      if (w)
        inputs.push_back(*w);
    }
    t->SetLastInputs(std::move(inputs));
  }
  t->Fire();
  return t->Id();
}

int Scheduler::RunToQuiescence() {
  int fired = 0;
  while (Step() != -1)
    ++fired;
  return fired;
}

} // namespace polykernel::dataflow
