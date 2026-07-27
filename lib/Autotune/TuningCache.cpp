//===- TuningCache.cpp - JSON tuning-cache (contract H) ---------*- C++ -*-===//
//
// llvm::json implementation of the contract-H tuning cache. ParseTuningCache is
// a strict parse-don't-validate decoder: every required field is fetched with an
// explicit type check, and the FIRST missing / wrong-typed field produces a
// schema error naming its full path (e.g. "entries[0].correctness.cosine").
// No field is ever silently defaulted. `time_ms` is the one nullable field:
// JSON null is valid (compile-time-model scoring) but the key must be present.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/TuningCache.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace json = llvm::json;

namespace polykernel::autotune {

namespace {

// ---- strict field accessors: each returns false + sets `error` on failure ----

/// Fetch a required sub-object. On success `out` points into `obj`.
bool GetObject(const json::Object &obj, llvm::StringRef key,
               const std::string &path, const json::Object *&out,
               std::string &error) {
  const json::Value *v = obj.get(key);
  if (v == nullptr) {
    error = path + "." + key.str() + ": missing required object field";
    return false;
  }
  out = v->getAsObject();
  if (out == nullptr) {
    error = path + "." + key.str() + ": expected an object";
    return false;
  }
  return true;
}

bool GetString(const json::Object &obj, llvm::StringRef key,
               const std::string &path, std::string &out, std::string &error) {
  const json::Value *v = obj.get(key);
  if (v == nullptr) {
    error = path + "." + key.str() + ": missing required string field";
    return false;
  }
  auto s = v->getAsString();
  if (!s) {
    error = path + "." + key.str() + ": expected a string";
    return false;
  }
  out = s->str();
  return true;
}

bool GetInt(const json::Object &obj, llvm::StringRef key, const std::string &path,
            long long &out, std::string &error) {
  const json::Value *v = obj.get(key);
  if (v == nullptr) {
    error = path + "." + key.str() + ": missing required integer field";
    return false;
  }
  auto i = v->getAsInteger();
  if (!i) {
    error = path + "." + key.str() + ": expected an integer";
    return false;
  }
  out = static_cast<long long>(*i);
  return true;
}

bool GetDouble(const json::Object &obj, llvm::StringRef key,
               const std::string &path, double &out, std::string &error) {
  const json::Value *v = obj.get(key);
  if (v == nullptr) {
    error = path + "." + key.str() + ": missing required number field";
    return false;
  }
  auto d = v->getAsNumber();
  if (!d) {
    error = path + "." + key.str() + ": expected a number";
    return false;
  }
  out = *d;
  return true;
}

bool GetBool(const json::Object &obj, llvm::StringRef key, const std::string &path,
             bool &out, std::string &error) {
  const json::Value *v = obj.get(key);
  if (v == nullptr) {
    error = path + "." + key.str() + ": missing required boolean field";
    return false;
  }
  auto b = v->getAsBoolean();
  if (!b) {
    error = path + "." + key.str() + ": expected a boolean";
    return false;
  }
  out = *b;
  return true;
}

/// `time_ms`: the key must be present; value is a number OR JSON null.
bool GetNullableTime(const json::Object &obj, const std::string &path,
                     std::optional<double> &out, std::string &error) {
  const json::Value *v = obj.get("time_ms");
  if (v == nullptr) {
    error = path + ".time_ms: missing required field "
                   "(use null for a compile-time-model score)";
    return false;
  }
  if (v->kind() == json::Value::Null) {
    out = std::nullopt; // legitimate: scored by a compile-time model.
    return true;
  }
  auto d = v->getAsNumber();
  if (!d) {
    error = path + ".time_ms: expected a number or null";
    return false;
  }
  out = *d;
  return true;
}

bool ParseShape(const json::Object &obj, const std::string &path, Shape &out,
                std::string &error) {
  long long m = 0, n = 0, k = 0;
  if (!GetInt(obj, "M", path, m, error))
    return false;
  if (!GetInt(obj, "N", path, n, error))
    return false;
  if (!GetInt(obj, "K", path, k, error))
    return false;
  if (!GetString(obj, "dtype", path, out.dtype, error))
    return false;
  out.m = m;
  out.n = n;
  out.k = k;
  return true;
}

bool ParseConfig(const json::Object &obj, const std::string &path, Config &out,
                 std::string &error) {
  long long bm = 0, bn = 0, bk = 0, nw = 0, vw = 0, un = 0, st = 0;
  if (!GetInt(obj, "block_m", path, bm, error))
    return false;
  if (!GetInt(obj, "block_n", path, bn, error))
    return false;
  if (!GetInt(obj, "block_k", path, bk, error))
    return false;
  if (!GetInt(obj, "num_warps", path, nw, error))
    return false;
  if (!GetInt(obj, "vector_width", path, vw, error))
    return false;
  if (!GetInt(obj, "unroll", path, un, error))
    return false;
  if (!GetInt(obj, "shared_memory_stages", path, st, error))
    return false;
  out = Config{static_cast<int>(bm), static_cast<int>(bn), static_cast<int>(bk),
               static_cast<int>(nw), static_cast<int>(vw), static_cast<int>(un),
               static_cast<int>(st)};
  return true;
}

bool ParseCorrectness(const json::Object &obj, const std::string &path,
                      Correctness &out, std::string &error) {
  if (!GetDouble(obj, "cosine", path, out.cosine, error))
    return false;
  if (!GetDouble(obj, "max_rel_err", path, out.max_rel_err, error))
    return false;
  if (!GetDouble(obj, "pcc", path, out.pcc, error))
    return false;
  return true;
}

bool ParseEntry(const json::Object &obj, const std::string &path,
                CacheEntry &out, std::string &error) {
  if (!GetString(obj, "gpu", path, out.gpu, error))
    return false;
  if (!GetString(obj, "op", path, out.op, error))
    return false;

  const json::Object *shape = nullptr;
  if (!GetObject(obj, "shape", path, shape, error))
    return false;
  if (!ParseShape(*shape, path + ".shape", out.shape, error))
    return false;

  const json::Object *best = nullptr;
  if (!GetObject(obj, "best", path, best, error))
    return false;
  if (!ParseConfig(*best, path + ".best", out.best, error))
    return false;

  if (!GetString(obj, "scored_by", path, out.scored_by, error))
    return false;
  if (!GetNullableTime(obj, path, out.time_ms, error))
    return false;
  if (!GetBool(obj, "validated", path, out.validated, error))
    return false;

  const json::Object *corr = nullptr;
  if (!GetObject(obj, "correctness", path, corr, error))
    return false;
  if (!ParseCorrectness(*corr, path + ".correctness", out.correctness, error))
    return false;
  return true;
}

// ---- serializers ----

json::Object ToJson(const Shape &s) {
  json::Object o;
  o.try_emplace("M", static_cast<int64_t>(s.m));
  o.try_emplace("N", static_cast<int64_t>(s.n));
  o.try_emplace("K", static_cast<int64_t>(s.k));
  o.try_emplace("dtype", s.dtype);
  return o;
}

json::Object ToJson(const Config &c) {
  json::Object o;
  o.try_emplace("block_m", c.block_m);
  o.try_emplace("block_n", c.block_n);
  o.try_emplace("block_k", c.block_k);
  o.try_emplace("num_warps", c.num_warps);
  o.try_emplace("vector_width", c.vector_width);
  o.try_emplace("unroll", c.unroll);
  o.try_emplace("shared_memory_stages", c.shared_memory_stages);
  return o;
}

json::Object ToJson(const Correctness &c) {
  json::Object o;
  o.try_emplace("cosine", c.cosine);
  o.try_emplace("max_rel_err", c.max_rel_err);
  o.try_emplace("pcc", c.pcc);
  return o;
}

json::Object ToJson(const CacheEntry &e) {
  json::Object o;
  o.try_emplace("gpu", e.gpu);
  o.try_emplace("op", e.op);
  o.try_emplace("shape", ToJson(e.shape));
  o.try_emplace("best", ToJson(e.best));
  o.try_emplace("scored_by", e.scored_by);
  // time_ms: nullopt => JSON null (compile-time-model score).
  if (e.time_ms.has_value())
    o.try_emplace("time_ms", *e.time_ms);
  else
    o.try_emplace("time_ms", json::Value(nullptr));
  o.try_emplace("validated", e.validated);
  o.try_emplace("correctness", ToJson(e.correctness));
  return o;
}

} // namespace

TuningCacheParseResult ParseTuningCache(std::string_view text) {
  TuningCacheParseResult result;

  auto parsed = json::parse(llvm::StringRef(text.data(), text.size()));
  if (!parsed) {
    result.error = "malformed JSON: " +
                   llvm::toString(parsed.takeError());
    return result;
  }

  const json::Object *root = parsed->getAsObject();
  if (root == nullptr) {
    result.error = "root: expected a JSON object (the versioned envelope)";
    return result;
  }

  TuningCache cache;
  long long version = 0;
  if (!GetInt(*root, "version", "root", version, result.error))
    return result;
  if (version != kTuningCacheVersion) {
    result.error = "root.version: unsupported schema version " +
                   std::to_string(version) + " (expected " +
                   std::to_string(kTuningCacheVersion) + ")";
    return result;
  }
  cache.version = static_cast<int>(version);

  const json::Value *entries_val = root->get("entries");
  if (entries_val == nullptr) {
    result.error = "root.entries: missing required array field";
    return result;
  }
  const json::Array *entries = entries_val->getAsArray();
  if (entries == nullptr) {
    result.error = "root.entries: expected an array";
    return result;
  }

  cache.entries.reserve(entries->size());
  for (std::size_t i = 0; i < entries->size(); ++i) {
    const json::Object *entry_obj = (*entries)[i].getAsObject();
    const std::string path = "entries[" + std::to_string(i) + "]";
    if (entry_obj == nullptr) {
      result.error = path + ": expected an object";
      return result;
    }
    CacheEntry entry;
    if (!ParseEntry(*entry_obj, path, entry, result.error))
      return result;
    cache.entries.push_back(std::move(entry));
  }

  result.cache = std::move(cache);
  return result;
}

std::string SerializeTuningCache(const TuningCache &cache) {
  json::Array entries;
  for (const CacheEntry &e : cache.entries)
    entries.push_back(json::Value(ToJson(e)));

  json::Object root;
  root.try_emplace("version", cache.version);
  root.try_emplace("entries", std::move(entries));

  std::string out;
  llvm::raw_string_ostream os(out);
  os << json::Value(std::move(root));
  return out;
}

} // namespace polykernel::autotune
