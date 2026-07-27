//===- KernelCache.cpp - (gpu,op,shape) -> best cached kernel ---*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). See KernelCache.h.
//
//===----------------------------------------------------------------------===//
//
// The cache is a thin selection layer over the contract-H AmdTuningDb (Todo 23):
// LoadTuningCache parses a contract-H JSON envelope (reusing ParseTuningCache, so
// any schema error propagates with its field path) and partitions the entries by
// gpu; Select looks up (gpu, op, shape) WITHIN the detected gpu namespace and
// gates on `validated`. The two graceful-miss errors (absent entry; present-but-
// unvalidated) are the ONLY outcomes besides a served validated entry - a wrong
// kernel is never returned and a miss never crashes.
//
//===----------------------------------------------------------------------===//

// allow: SIZE_OK - one cohesive cache class: selection (Todo 26) + its persistence /
// hash-invalidation / multi-GPU hardening (Todo 42) are a single responsibility. The
// persistence I/O + binary-metadata JSON cannot move to a sibling TU without editing
// lib/Runtime/CMakeLists.txt (out of scope for T42, which may touch only the tests
// CMake). The contract-H tuning parse is REUSED (ParseTuningCache), not reimplemented.

#include "PolyKernel/Runtime/KernelCache.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <set>
#include <system_error>

namespace json = llvm::json;

namespace polykernel::runtime {

std::string FormatKey(std::string_view gpu, std::string_view op,
                      const Shape &shape) {
  return std::string(gpu) + "/" + std::string(op) + "/M=" +
         std::to_string(shape.m) + ",N=" + std::to_string(shape.n) + ",K=" +
         std::to_string(shape.k) + "," + shape.dtype;
}

bool HashFile(std::string_view so_path, std::string &hash_out,
              std::string &error) {
  auto buf = llvm::MemoryBuffer::getFile(std::string(so_path));
  if (!buf) {
    error = "cannot read binary " + std::string(so_path) + ": " +
            buf.getError().message();
    return false;
  }
  // FNV-1a 64-bit over the raw bytes: stable across runs + platforms, so the same
  // .so always hashes identically and any byte change flips the hash.
  std::uint64_t h = 1469598103934665603ULL; // FNV offset basis.
  for (const char c : (*buf)->getBuffer()) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ULL; // FNV prime.
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string hex(16, '0');
  for (int i = 15; i >= 0; --i) {
    hex[static_cast<std::size_t>(i)] = kHex[h & 0xFULL];
    h >>= 4U;
  }
  hash_out = std::move(hex);
  return true;
}

bool KernelCache::LoadTuningCache(std::string_view json, std::string &error) {
  autotune::AmdTuningDbParseResult parsed = autotune::AmdTuningDb::Parse(json);
  if (!parsed.ok()) {
    error = std::move(parsed.error);
    return false;
  }
  db_ = std::move(*parsed.db);
  return true;
}

bool KernelCache::Put(CacheEntry entry) { return db_.Put(std::move(entry)); }

SelectionResult KernelCache::Select(std::string_view gpu, std::string_view op,
                                    const Shape &shape) const {
  SelectionResult result;
  const std::optional<CacheEntry> entry = db_.Get(gpu, op, shape);
  if (!entry.has_value()) {
    // The pinned negative path: NO cache entry for this (gpu, op, shape). Return a
    // clear, actionable error - never a wrong kernel, never a crash.
    result.error =
        "no tuned kernel for " + FormatKey(gpu, op, shape) +
        "; run the autotuner (polykernel-bench --autotune) to populate the cache";
    return result;
  }
  if (!entry->validated) {
    // An entry exists but never passed the correctness gate (validated:false). Do
    // NOT serve it - a bench-rejected variant is treated as a miss.
    result.error = "cached entry for " + FormatKey(gpu, op, shape) +
                   " is not validated; run the autotuner to produce a "
                   "correctness-gated variant";
    return result;
  }
  result.entry = entry;
  return result;
}

// ---- Hardening (Todo 42): binary metadata + persist + invalidate + multi-GPU ----

namespace {

constexpr int kPersistVersion = 1;
constexpr const char *kTuningDbFile = "/tuning_db.json";
constexpr const char *kBinariesFile = "/binaries.json";

json::Object ToJson(const Shape &s) {
  json::Object o;
  o.try_emplace("M", static_cast<std::int64_t>(s.m));
  o.try_emplace("N", static_cast<std::int64_t>(s.n));
  o.try_emplace("K", static_cast<std::int64_t>(s.k));
  o.try_emplace("dtype", s.dtype);
  return o;
}

// Strict field accessors (field-path errors; mirror the contract-H parser so a
// malformed binaries.json is rejected with the offending path, never defaulted).
bool GetStr(const json::Object &obj, llvm::StringRef key, const std::string &path,
            std::string &out, std::string &error) {
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

bool GetI64(const json::Object &obj, llvm::StringRef key, const std::string &path,
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

bool GetObj(const json::Object &obj, llvm::StringRef key, const std::string &path,
            const json::Object *&out, std::string &error) {
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

bool ParseShape(const json::Object &obj, const std::string &path, Shape &out,
                std::string &error) {
  long long m = 0, n = 0, k = 0;
  if (!GetI64(obj, "M", path, m, error))
    return false;
  if (!GetI64(obj, "N", path, n, error))
    return false;
  if (!GetI64(obj, "K", path, k, error))
    return false;
  if (!GetStr(obj, "dtype", path, out.dtype, error))
    return false;
  out.m = m;
  out.n = n;
  out.k = k;
  return true;
}

bool ReadFile(const std::string &path, std::string &out, std::string &error) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) {
    error = "cannot read " + path + ": " + buf.getError().message();
    return false;
  }
  out = (*buf)->getBuffer().str();
  return true;
}

bool WriteFile(const std::string &path, const std::string &content,
               std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot open " + path + " for writing: " + ec.message();
    return false;
  }
  os << content;
  os.flush();
  if (os.has_error()) {
    error = "write failed for " + path;
    return false;
  }
  return true;
}

// binaries.json envelope: {version, binaries:[{gpu,op,shape,so_path,kernel_hash}]}.
// The structured key (gpu/op/shape) comes from the tuning entry each binary backs.
std::string SerializeBinaries(const autotune::AmdTuningDb &db,
                              const std::map<std::string, CachedBinary> &bins) {
  json::Array arr;
  for (const std::string &ns : db.Namespaces()) {
    for (const CacheEntry &e : db.Entries(ns)) {
      const auto it = bins.find(FormatKey(e.gpu, e.op, e.shape));
      if (it == bins.end())
        continue; // a tuning entry with no recorded binary is not persisted here.
      json::Object o;
      o.try_emplace("gpu", e.gpu);
      o.try_emplace("op", e.op);
      o.try_emplace("shape", ToJson(e.shape));
      o.try_emplace("so_path", it->second.so_path);
      o.try_emplace("kernel_hash", it->second.kernel_hash);
      arr.push_back(json::Value(std::move(o)));
    }
  }
  json::Object root;
  root.try_emplace("version", kPersistVersion);
  root.try_emplace("binaries", std::move(arr));
  std::string out;
  llvm::raw_string_ostream os(out);
  os << json::Value(std::move(root));
  return out;
}

// Strict parse of binaries.json into `out` (keyed by FormatKey). Every binary record
// must back a present tuning entry in `db` (consistency; parse-don't-validate). On
// failure sets `error` to an UNPREFIXED field path; the caller adds "binaries.json:".
bool ParseBinaries(std::string_view text, const autotune::AmdTuningDb &db,
                   std::map<std::string, CachedBinary> &out, std::string &error) {
  auto parsed = json::parse(llvm::StringRef(text.data(), text.size()));
  if (!parsed) {
    error = "malformed JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  const json::Object *root = parsed->getAsObject();
  if (root == nullptr) {
    error = "root: expected a JSON object (the binaries envelope)";
    return false;
  }
  long long version = 0;
  if (!GetI64(*root, "version", "root", version, error))
    return false;
  if (version != kPersistVersion) {
    error = "root.version: unsupported binaries version " + std::to_string(version) +
            " (expected " + std::to_string(kPersistVersion) + ")";
    return false;
  }
  const json::Value *bv = root->get("binaries");
  if (bv == nullptr) {
    error = "root.binaries: missing required array field";
    return false;
  }
  const json::Array *arr = bv->getAsArray();
  if (arr == nullptr) {
    error = "root.binaries: expected an array";
    return false;
  }
  for (std::size_t i = 0; i < arr->size(); ++i) {
    const std::string path = "binaries[" + std::to_string(i) + "]";
    const json::Object *o = (*arr)[i].getAsObject();
    if (o == nullptr) {
      error = path + ": expected an object";
      return false;
    }
    std::string gpu, op, so_path, kernel_hash;
    Shape shape;
    if (!GetStr(*o, "gpu", path, gpu, error))
      return false;
    if (!GetStr(*o, "op", path, op, error))
      return false;
    const json::Object *shape_obj = nullptr;
    if (!GetObj(*o, "shape", path, shape_obj, error))
      return false;
    if (!ParseShape(*shape_obj, path + ".shape", shape, error))
      return false;
    if (!GetStr(*o, "so_path", path, so_path, error))
      return false;
    if (!GetStr(*o, "kernel_hash", path, kernel_hash, error))
      return false;
    if (!db.Get(gpu, op, shape).has_value()) {
      error = path + ": no tuning entry for " + FormatKey(gpu, op, shape) +
              " (a binary must back a present tuning entry)";
      return false;
    }
    out[FormatKey(gpu, op, shape)] = CachedBinary{std::move(so_path),
                                                  std::move(kernel_hash)};
  }
  return true;
}

} // namespace

bool KernelCache::PutBinary(std::string_view gpu, std::string_view op,
                            const Shape &shape, CachedBinary binary) {
  if (gpu.empty())
    return false;
  if (!db_.Get(gpu, op, shape).has_value())
    return false; // a binary must back a real, present tuning entry.
  binaries_[FormatKey(gpu, op, shape)] = std::move(binary);
  return true;
}

std::optional<CachedBinary>
KernelCache::GetBinary(std::string_view gpu, std::string_view op,
                       const Shape &shape) const {
  const auto it = binaries_.find(FormatKey(gpu, op, shape));
  if (it == binaries_.end())
    return std::nullopt;
  return it->second;
}

bool KernelCache::Persist(std::string_view cache_dir, std::string &error) const {
  const std::string dir(cache_dir);
  if (const std::error_code ec = llvm::sys::fs::create_directories(dir); ec) {
    error = "cannot create cache dir " + dir + ": " + ec.message();
    return false;
  }
  if (!WriteFile(dir + kTuningDbFile, db_.Serialize(), error))
    return false;
  if (!WriteFile(dir + kBinariesFile, SerializeBinaries(db_, binaries_), error))
    return false;
  return true;
}

bool KernelCache::LoadPersisted(std::string_view cache_dir, std::string &error) {
  const std::string dir(cache_dir);
  std::string tuning_json, binaries_json;
  if (!ReadFile(dir + kTuningDbFile, tuning_json, error))
    return false;
  if (!ReadFile(dir + kBinariesFile, binaries_json, error))
    return false;

  // Parse into TEMPORARIES first: the cache is left UNCHANGED on any failure
  // (parse-don't-validate; no partial load, no silent default).
  autotune::AmdTuningDbParseResult parsed =
      autotune::AmdTuningDb::Parse(tuning_json);
  if (!parsed.ok()) {
    error = "tuning_db.json: " + parsed.error; // contract-H field path intact.
    return false;
  }
  std::map<std::string, CachedBinary> bins;
  if (!ParseBinaries(binaries_json, *parsed.db, bins, error)) {
    error = "binaries.json: " + error;
    return false;
  }
  db_ = std::move(*parsed.db);
  binaries_ = std::move(bins);
  return true;
}

std::size_t KernelCache::InvalidateStale() {
  std::set<std::string> stale;
  for (const auto &[key, bin] : binaries_) {
    std::string hash, hash_error;
    if (!HashFile(bin.so_path, hash, hash_error) || hash != bin.kernel_hash)
      stale.insert(key); // missing/unreadable .so OR hash mismatch => stale.
  }
  if (stale.empty())
    return 0;

  // Rebuild the tuning DB without the stale entries (the AmdTuningDb exposes no
  // erase; reconstruction via its public namespace/entry iteration is the removal).
  autotune::AmdTuningDb fresh;
  for (const std::string &ns : db_.Namespaces())
    for (const CacheEntry &e : db_.Entries(ns))
      if (!stale.contains(FormatKey(e.gpu, e.op, e.shape)))
        fresh.Put(e);
  db_ = std::move(fresh);
  for (const std::string &key : stale)
    binaries_.erase(key);
  return stale.size();
}

std::vector<std::pair<std::string, SelectionResult>>
KernelCache::SelectPerGpu(const std::vector<std::string> &gpus,
                          std::string_view op, const Shape &shape) const {
  std::vector<std::pair<std::string, SelectionResult>> out;
  out.reserve(gpus.size());
  for (const std::string &gpu : gpus)
    out.emplace_back(gpu, Select(gpu, op, shape));
  return out;
}

} // namespace polykernel::runtime
