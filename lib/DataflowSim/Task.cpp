//===- Task.cpp - Dataflow task model --------------------------*- C++ -*-===//
//
// PolyKernel dataflow task model + scheduler (Todo 36 / Wave 7). See Task.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Task.h"

namespace polykernel::dataflow {

const char *TaskKindName(TaskKind k) {
  switch (k) {
  case TaskKind::Data:
    return "data";
  case TaskKind::Local:
    return "local";
  case TaskKind::Control:
    return "control";
  }
  return "unknown";
}

Task::Task(int id, TaskKind kind) : id_(id), kind_(kind) {}

void Task::AddInput(Color c) { inputs_.push_back(c); }

void Task::Fire() {
  ++fire_count_;
  if (handler_)
    handler_(*this);
}

} // namespace polykernel::dataflow
