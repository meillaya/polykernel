# -*- Python -*-
#
# PolyKernel out-of-tree lit configuration (W1 spike). Modeled on
# mlir/examples/standalone/test/lit.cfg.py, trimmed to the dialect spike.

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "POLYKERNEL"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.polykernel_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])

# excludes: auxiliary-input subdirs + build files are not tests.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# Tool resolution:
#   polykernel-opt  -> this build (build/tools/polykernel-opt)
#   FileCheck/count/not -> installed LLVM tools dir (nix llvmPackages_21.llvm)
config.polykernel_tools_dir = os.path.join(
    config.polykernel_obj_root, "tools", "polykernel-opt"
)

tool_dirs = [config.polykernel_tools_dir, config.llvm_tools_dir]
tools = ["polykernel-opt", "FileCheck", "count", "not"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
