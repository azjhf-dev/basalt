// The router's geometry contract (include/basalt/models/deepseek41/kernels.hpp): which RouterArgs router() takes and
// which it throws on. The point of the contract: a pruned model's expert count (200 of the published 384, say) is not
// a multiple of 32, and the selection block pads to whole warps instead of refusing it. Host-only: no CUDA.
#include "basalt/models/deepseek41/kernels.hpp"

#include <cstdio>

using namespace basalt::models::deepseek41;

namespace {

int failures = 0;

void expect(bool cond, const char* what, const RouterArgs& a) {
    if (cond) return;
    std::printf("router_geometry_test: %s (n_expert=%d n_embd=%d k=%d)\n", what, a.n_expert, a.n_embd, a.k);
    ++failures;
}

RouterArgs args(int n_expert, int n_embd = 5120, int k = 6) {
    RouterArgs a;
    a.n_expert = n_expert;
    a.n_embd = n_embd;
    a.k = k;
    return a;
}

}  // namespace

int main() {
    // the shipped model and the pruned quant (200 of 384 experts kept)
    expect(router_geometry_ok(RouterArgs{}), "the default geometry is refused", RouterArgs{});
    expect(router_geometry_ok(args(200)), "a pruned model's 200 experts are refused", args(200));
    expect(router_geometry_ok(args(96, 128, 1)), "the smallest whole-warp-multiple geometry is refused",
           args(96, 128, 1));

    // every count the padding claims to take, at both k ends
    for (int n = 1; n <= kRouterMaxExpert; ++n) {
        expect(router_geometry_ok(args(n, 5120, 1)), "an expert count in range is refused", args(n, 5120, 1));
        const int k = n < kRouterMaxK ? n : kRouterMaxK;
        expect(router_geometry_ok(args(n, 5120, k)), "an expert count in range is refused at max k", args(n, 5120, k));
    }

    // the refusals
    expect(!router_geometry_ok(args(0)), "zero experts taken", args(0));
    expect(!router_geometry_ok(args(kRouterMaxExpert + 1)), "too many experts taken", args(kRouterMaxExpert + 1));
    expect(!router_geometry_ok(args(-32)), "a negative expert count taken", args(-32));
    expect(!router_geometry_ok(args(384, 0)), "a zero embedding taken", args(384, 0));
    expect(!router_geometry_ok(args(384, 96)), "an embedding under one tile taken", args(384, 96));
    expect(!router_geometry_ok(args(384, 5120 + 64)), "a non-128-multiple embedding taken", args(384, 5120 + 64));
    expect(!router_geometry_ok(args(384, kRouterMaxEmbd + 128)), "too wide an embedding taken",
           args(384, kRouterMaxEmbd + 128));
    expect(!router_geometry_ok(args(384, 5120, 0)), "k = 0 taken", args(384, 5120, 0));
    expect(!router_geometry_ok(args(384, 5120, kRouterMaxK + 1)), "too large a k taken", args(384, 5120, kRouterMaxK + 1));
    expect(!router_geometry_ok(args(4, 5120, 6)), "k above the expert count taken", args(4, 5120, 6));
    expect(router_geometry_ok(args(4, 5120, 4)), "k == n_expert refused", args(4, 5120, 4));

    std::printf(failures ? "router_geometry_test: FAILED\n" : "router_geometry_test: ok\n");
    return failures ? 1 : 0;
}
