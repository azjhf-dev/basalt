#!/bin/sh
# basalt-mmap-launcher.sh - launch bin/basalt with the glibc allocator arena
# workaround from Robert Fairburn's owner-local mmap4K recipe (MIT), branch
# experiment/owner-local-mmap4k-primary, commit 30ed20d41a4a7c8a5e61d6988035146528322f04
# (rdna3/recipes/owner-local-mmap4k/native-exec.sh.in), adapted for Basalt.
#
# What it does, and why:
#   MALLOC_MMAP_THRESHOLD_=131072 encourages glibc to serve large allocations
#   with mmap, so their pages go back to the OS on free instead of sitting
#   inside allocator arenas, where the OS-headroom gate cannot see them. This
#   is a workaround for a retention symptom on long-running loads, not a
#   stability fix or a claim that mmap is faster by default.
#   It also zeroes the child's core-dump limits and unsets PWD, as the
#   original wrapper does, and refuses to run with LD_PRELOAD set.
#   Basalt reads no profiling environment variables (checked against
#   e9db1aea), so there is nothing to suppress here beyond PWD.
#
# UNTESTED ON HARDWARE. The original recipe was qualified on 4x RX 7900 XTX
# (ROCm); nobody has measured this wrapper with Basalt on a Blackwell card.
# Run your normal path without it first and keep it as the fallback.
#
# Usage: fixes/basalt-mmap-launcher.sh /ABS/PATH/bin/basalt serve --config /ABS/PATH/basalt_config.json
# (or any other basalt invocation; arguments pass through unchanged)
[ "${LD_PRELOAD+x}" != x ] || exit 125
[ "${1-}" != "" ] || exit 125
BIN=$1; shift
[ -x "$BIN" ] || exit 125
ulimit -S -c 0 || exit 125
ulimit -H -c 0 || exit 125
unset PWD || exit 125
MALLOC_MMAP_THRESHOLD_=131072 exec "$BIN" "$@"
