// include/basalt/prefill/dp_plan.hpp - data-parallel chunk plan for one prompt across several GPU paths.
//
// Ported from Robert Fairburn's Strata-rdna3 fork (MIT), branch
// build/rdna3-f557b54-p4-fixes, commit b7c5a2644c5ee0e15f67b1d3edd40b09ae92eb69
// (include/strata/prefill/dp_plan.hpp), carried on Niko1221/Strata (MIT). Only the
// namespace changed. See fixes/README.md for what this is for and why the engine
// does not call it yet.
//
// The planner splits one prompt of n tokens into chunks of at most cap tokens,
// 256-aligned, dealt round-robin over `paths` GPU paths so every path gets one
// chunk before any gets a second, and the last chunk still holds at least
// min_last tokens (a short tail would not pay for its own pass). Pure: no GPU,
// no environment.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace basalt::prefill::dp {

struct Chunk {
    int64_t c0 = 0, T = 0, k = 0;
    int who = 0;
};

inline int64_t step_for(int64_t n, int64_t cap, int paths) {
    if (n <= 0 || cap <= 0 || paths < 2) return 0;
    const int64_t limit = std::numeric_limits<int64_t>::max();
    int64_t count = std::max<int64_t>(paths, n / cap + (n % cap != 0));
    const int64_t extra = (paths - count % paths) % paths;
    if (count > limit - extra) return 0;
    count += extra;
    const int64_t tokens = n / count + (n % count != 0);
    if (tokens > limit - 255) return 0;
    return (tokens + 255) / 256 * 256;
}

inline bool takes(int64_t n, int64_t cap, int paths, int64_t min_last) {
    const int64_t step = step_for(n, cap, paths);
    if (step <= 0 || step > cap) return false;
    const int64_t count = n / step + (n % step != 0);
    return count >= paths && n - (count - 1) * step >= min_last;
}

inline std::vector<Chunk> plan(int64_t n, int64_t cap, int paths, int64_t min_last) {
    if (!takes(n, cap, paths, min_last)) return {};
    const int64_t step = step_for(n, cap, paths);
    const int64_t count = n / step + (n % step != 0);
    std::vector<Chunk> out;
    out.reserve((size_t) count);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t begin = i * step;
        const int who = paths - 1 - (int) ((count - 1 - i) % paths);
        out.push_back({begin, std::min(step, n - begin), i, who});
    }
    return out;
}

inline int64_t handoff_begin(const std::vector<Chunk>& chunks, int64_t k, int paths) {
    const int64_t first = std::max<int64_t>(0, k - paths + 2);
    return chunks[(size_t) first].c0;
}

}  // namespace basalt::prefill::dp
