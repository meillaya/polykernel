//===- npy_io.cpp - Minimal NumPy .npy I/O for bf16 (<V2>) -------*- C++ -*-===//
//
// PolyKernel CPU reference .npy bridge (Todo 8 / Wave 2). See npy_io.h.
//
//===----------------------------------------------------------------------===//
//
// Byte order: the target is x86_64-linux (little-endian), matching the '<V2'
// little-endian descr, so the raw uint16 payload is read/written directly with
// no byte swap.
//
//===----------------------------------------------------------------------===//

#include "npy_io.h"

#include <cstdint>
#include <fstream>

namespace polykernel::cpu {

namespace {

constexpr char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

int64_t product(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

/// Extract the single-quoted string value following `key` in the header dict
/// (e.g. key="'descr'" -> "<V2"). Throws NpyError if not found.
std::string quoted_value_after(const std::string &header, const std::string &key) {
  std::string::size_type k = header.find(key);
  if (k == std::string::npos)
    throw NpyError("npy header missing key " + key);
  std::string::size_type open = header.find('\'', k + key.size());
  if (open == std::string::npos)
    throw NpyError("npy header: no opening quote for " + key);
  std::string::size_type close = header.find('\'', open + 1);
  if (close == std::string::npos)
    throw NpyError("npy header: no closing quote for " + key);
  return header.substr(open + 1, close - open - 1);
}

/// Parse the `shape` tuple (e.g. "(2, 3)" / "(3,)" / "()") following the
/// 'shape' key. Throws NpyError if malformed.
std::vector<int64_t> parse_shape(const std::string &header) {
  std::string::size_type k = header.find("'shape'");
  if (k == std::string::npos)
    throw NpyError("npy header missing 'shape'");
  std::string::size_type open = header.find('(', k);
  std::string::size_type close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos)
    throw NpyError("npy header: malformed 'shape' tuple");

  std::vector<int64_t> shape;
  std::string::size_type i = open + 1;
  while (i < close) {
    while (i < close && (header[i] == ' ' || header[i] == ','))
      ++i; // skip separators / whitespace (incl. the trailing comma of a 1-tuple)
    if (i >= close)
      break;
    std::string::size_type start = i;
    while (i < close && header[i] >= '0' && header[i] <= '9')
      ++i;
    if (i == start)
      throw NpyError("npy header: non-integer dimension in 'shape'");
    shape.push_back(std::stoll(header.substr(start, i - start)));
  }
  return shape;
}

} // namespace

int64_t NpyArray::size() const { return product(shape); }

NpyArray read_npy_bf16(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw NpyError("cannot open npy file for reading: " + path);

  char magic[6];
  in.read(magic, 6);
  if (!in || std::string(magic, 6) != std::string(kMagic, 6))
    throw NpyError("not a .npy file (bad magic): " + path);

  unsigned char ver[2];
  in.read(reinterpret_cast<char *>(ver), 2);

  // v1.0: 2-byte little-endian header length; v2.0+: 4-byte little-endian.
  uint64_t header_len = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    in.read(reinterpret_cast<char *>(b), 2);
    header_len = static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8);
  } else {
    unsigned char b[4];
    in.read(reinterpret_cast<char *>(b), 4);
    header_len = static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8) |
                 (static_cast<uint64_t>(b[2]) << 16) |
                 (static_cast<uint64_t>(b[3]) << 24);
  }
  if (!in)
    throw NpyError("truncated .npy header: " + path);

  std::string header(static_cast<std::size_t>(header_len), '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in)
    throw NpyError("truncated .npy header body: " + path);

  std::string descr = quoted_value_after(header, "'descr'");
  if (descr != "<V2" && descr != "|V2")
    throw NpyError("unsupported npy descr '" + descr + "' (expected '<V2' bf16)");

  NpyArray arr;
  arr.shape = parse_shape(header);
  int64_t n = arr.size();
  arr.data.resize(static_cast<std::size_t>(n));
  in.read(reinterpret_cast<char *>(arr.data.data()),
          static_cast<std::streamsize>(n * 2));
  if (!in)
    throw NpyError("truncated .npy payload: " + path);
  return arr;
}

void write_npy_bf16(const std::string &path, const std::vector<int64_t> &shape,
                    const std::vector<uint16_t> &data) {
  if (product(shape) != static_cast<int64_t>(data.size()))
    throw NpyError("npy write: data size != product(shape)");

  // Shape tuple in NumPy's repr form: "(2, 3)", "(3,)" for a 1-tuple, "()".
  std::string shape_str = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i)
      shape_str += ", ";
    shape_str += std::to_string(shape[i]);
  }
  if (shape.size() == 1)
    shape_str += ",";
  shape_str += ")";

  std::string dict = "{'descr': '<V2', 'fortran_order': False, 'shape': " +
                     shape_str + ", }";

  // v1.0: magic(6) + version(2) + header_len(2) + header is a multiple of 64.
  const std::size_t prefix = 10;
  std::size_t total_unpadded = prefix + dict.size() + 1; // +1 for trailing '\n'
  std::size_t padded_total = ((total_unpadded + 63) / 64) * 64;
  std::size_t header_len = padded_total - prefix;
  std::string header = dict;
  header.append(header_len - dict.size() - 1, ' ');
  header.push_back('\n');

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw NpyError("cannot open npy file for writing: " + path);

  out.write(kMagic, 6);
  const unsigned char ver[2] = {1, 0};
  out.write(reinterpret_cast<const char *>(ver), 2);
  const unsigned char len[2] = {
      static_cast<unsigned char>(header_len & 0xFF),
      static_cast<unsigned char>((header_len >> 8) & 0xFF)};
  out.write(reinterpret_cast<const char *>(len), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size() * 2));
  if (!out)
    throw NpyError("failed writing .npy payload: " + path);
}

} // namespace polykernel::cpu
