//===- AmdTuningDb.cpp - Per-GPU-namespaced AMD tuning DB -------*- C++ -*-===//
//
// Implementation of the AMD tuning database (Todo 23). A thin partitioning
// layer over the contract-H TuningCache (Todo 24): entries are keyed by
// (gpu, op, shape) in a std::map whose LEADING key component is the gpu, so
// each gpu is a contiguous namespace and gfx1101 / gfx942 never collide.
// Serialization delegates to SerializeTuningCache / ParseTuningCache - there is
// NO separate JSON schema; the namespace is carried by each entry's gpu field.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/AmdTuningDb.h"

#include <utility>

namespace polykernel::autotune {

bool AmdTuningDb::Put(CacheEntry entry) {
  if (entry.gpu.empty())
    return false; // an entry must belong to a gpu namespace.
  Key key{entry.gpu, entry.op, entry.shape};
  entries_[std::move(key)] = std::move(entry);
  return true;
}

std::optional<CacheEntry> AmdTuningDb::Get(std::string_view gpu,
                                           std::string_view op,
                                           const Shape &shape) const {
  const auto it = entries_.find(Key{std::string(gpu), std::string(op), shape});
  if (it == entries_.end())
    return std::nullopt;
  return it->second;
}

std::vector<CacheEntry> AmdTuningDb::Entries(std::string_view gpu) const {
  std::vector<CacheEntry> out;
  for (const auto &[key, entry] : entries_)
    if (key.gpu == gpu)
      out.push_back(entry);
  return out;
}

std::vector<std::string> AmdTuningDb::Namespaces() const {
  // The map is ordered gpu-first, so one namespace is a contiguous run: a
  // namespace appears exactly when the gpu differs from the previous key's.
  std::vector<std::string> out;
  for (const auto &[key, entry] : entries_)
    if (out.empty() || out.back() != key.gpu)
      out.push_back(key.gpu);
  return out;
}

std::string AmdTuningDb::Serialize() const {
  TuningCache cache;
  cache.entries.reserve(entries_.size());
  for (const auto &[key, entry] : entries_)
    cache.entries.push_back(entry);
  return SerializeTuningCache(cache);
}

std::string AmdTuningDb::SerializeNamespace(std::string_view gpu) const {
  TuningCache cache;
  for (const auto &[key, entry] : entries_)
    if (key.gpu == gpu)
      cache.entries.push_back(entry);
  return SerializeTuningCache(cache);
}

AmdTuningDbParseResult AmdTuningDb::Parse(std::string_view json) {
  AmdTuningDbParseResult result;

  auto parsed = ParseTuningCache(json);
  if (!parsed.ok()) {
    result.error = parsed.error; // contract-H schema error, field path intact.
    return result;
  }

  AmdTuningDb db;
  const auto &entries = parsed.cache->entries;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (!db.Put(entries[i])) {
      result.error = "entries[" + std::to_string(i) +
                     "].gpu: empty gpu - a tuning entry must belong to a "
                     "namespace (e.g. gfx1101, gfx942)";
      return result;
    }
  }

  result.db = std::move(db);
  return result;
}

} // namespace polykernel::autotune
