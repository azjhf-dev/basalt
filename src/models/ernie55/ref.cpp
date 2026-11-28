// src/models/ernie55/ref.cpp - see ref.hpp for the semantics this implements and the assumptions it flags.
#include "models/ernie55/ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace basalt::models::ernie55::ref {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

Vec matvec(const Mat& m, const float* x) {
    Vec out(m.rows, 0.f);
    for (int r = 0; r < m.rows; ++r) {
        const float* w = m.row(r);
        double acc = 0.0;
        for (int c = 0; c < m.cols; ++c) acc += (double) w[c] * x[c];
        out[r] = (float) acc;
    }
    return out;
}

float dot(const float* a, const float* b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) a[i] * b[i];
    return (float) acc;
}

void rmsnorm_inplace(Vec& x, const Vec* w, float eps) {
    double ss = 0.0;
    for (float v : x) ss += (double) v * v;
    const float inv = 1.f / (float) std::sqrt(ss / x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * inv * (w ? (*w)[i] : 1.f);
}

void layernorm_inplace(Vec& x, const Vec& w, const Vec& b, float eps) {
    double mean = 0.0;
    for (float v : x) mean += v;
    mean /= x.size();
    double var = 0.0;
    for (float v : x) var += ((double) v - mean) * ((double) v - mean);
    var /= x.size();
    const float inv = 1.f / (float) std::sqrt(var + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((x[i] - (float) mean) * inv) * w[i] + b[i];
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

/// Softmax over `scores` with one extra sink logit that only fattens the denominator (attention sinks).
Vec softmax_with_sink(const Vec& scores, float sink) {
    float mx = sink;
    for (float s : scores) mx = std::max(mx, s);
    double den = std::exp((double) sink - mx);
    Vec out(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        out[i] = (float) std::exp((double) scores[i] - mx);
        den += out[i];
    }
    for (float& v : out) v = (float) (v / den);
    return out;
}

RopeParams hca_rope(const Geometry& g) {
    RopeParams p;
    p.rot = g.rope_dim;
    p.base = g.comp_rope_base;
    p.freq_scale = 1.f / g.yarn_factor;
    p.n_ctx_orig = g.yarn_orig_ctx;
    p.beta_fast = g.yarn_beta_fast;
    p.beta_slow = g.yarn_beta_slow;
    return p;
}

RopeParams plain_rope(const Geometry& g) {
    RopeParams p;
    p.rot = g.rope_dim;
    p.base = g.rope_base;
    return p;
}

/// out[h] = sum_g mix[h*n + g] * in[g] over the head axis, per head_dim element (build_head_mix).
void head_mix(std::vector<Vec>& heads, const Vec& mix) {
    const int n = (int) heads.size();
    std::vector<Vec> out(n, Vec(heads[0].size(), 0.f));
    for (int h = 0; h < n; ++h)
        for (int g2 = 0; g2 < n; ++g2) {
            const float m = mix[h * n + g2];
            for (size_t d = 0; d < heads[h].size(); ++d) out[h][d] += m * heads[g2][d];
        }
    heads.swap(out);
}

bool shape_ok(const Mat& m, int rows, int cols, const std::string& name, std::string& err) {
    if (m.rows != rows || m.cols != cols) {
        err = name + ": shape [" + std::to_string(m.rows) + ", " + std::to_string(m.cols) + "] != [" +
              std::to_string(rows) + ", " + std::to_string(cols) + "] (the pack's meta and shapes disagree)";
        return false;
    }
    return true;
}

bool shape_ok(const Vec& v, int n, const std::string& name, std::string& err) {
    if ((int) v.size() != n) {
        err = name + ": " + std::to_string(v.size()) + " values != " + std::to_string(n);
        return false;
    }
    return true;
}

bool get_mat(const std::string& path, const Pack& p, const std::string& name, Mat& m, std::string& err) {
    const TensorRef* t = p.find(name);
    if (!t) {
        err = "pack has no tensor " + name;
        return false;
    }
    return read_mat(path, *t, m, err);
}

bool get_vec(const std::string& path, const Pack& p, const std::string& name, Vec& v, std::string& err) {
    const TensorRef* t = p.find(name);
    if (!t) {
        err = "pack has no tensor " + name;
        return false;
    }
    return read_vec(path, *t, v, err);
}

}  // namespace

// ------------------------------------------------------------ dequant

float fp16_to_f32(uint16_t h) {
    const int s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = (float) std::ldexp(m, -24);
    else if (e == 31) v = m ? std::numeric_limits<float>::quiet_NaN() : kInf;
    else v = (float) std::ldexp(m + 1024, e - 25);
    return s ? -v : v;
}

float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

namespace {

bool read_bytes(const std::string& path, const TensorRef& t, std::vector<char>& buf, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    f.seekg((std::streamoff) t.offset);
    buf.resize(t.bytes);
    f.read(buf.data(), (std::streamsize) t.bytes);
    if (!f) {
        err = "short read on " + t.name;
        return false;
    }
    return true;
}

}  // namespace

bool read_mat(const std::string& path, const TensorRef& t, Mat& out, std::string& err) {
    if (t.shape.size() != 2) {
        err = t.name + ": expected a 2D tensor, ne has " + std::to_string(t.shape.size()) + " dims";
        return false;
    }
    // pack order is fastest-first: ne [in, out] of the checkpoint's [out, in], data row-major [out][in]
    out.cols = (int) t.shape[0];
    out.rows = (int) t.shape[1];
    const size_t n = (size_t) out.rows * out.cols;
    std::vector<char> buf;
    if (!read_bytes(path, t, buf, err)) return false;
    out.w.resize(n);
    if (t.type == kTypeF32) {
        if (t.bytes != 4 * n) {
            err = t.name + ": F32 byte count mismatch";
            return false;
        }
        std::memcpy(out.w.data(), buf.data(), 4 * n);
    } else if (t.type == kTypeBF16) {
        if (t.bytes != 2 * n) {
            err = t.name + ": BF16 byte count mismatch";
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint16_t u;
            std::memcpy(&u, buf.data() + 2 * i, 2);
            out.w[i] = bf16_to_f32(u);
        }
    } else if (t.type == kTypeQ8_0) {
        if (n % 32 || t.bytes != n / 32 * 34) {
            err = t.name + ": Q8_0 byte count mismatch";
            return false;
        }
        for (size_t b = 0; b < n / 32; ++b) {
            uint16_t h;
            std::memcpy(&h, buf.data() + b * 34, 2);
            const float s = fp16_to_f32(h);
            for (int i = 0; i < 32; ++i) out.w[b * 32 + i] = s * (float) (int8_t) buf[b * 34 + 2 + i];
        }
    } else {
        err = t.name + ": ref reader cannot dequantize type " + std::to_string(t.type);
        return false;
    }
    return true;
}

bool read_vec(const std::string& path, const TensorRef& t, Vec& out, std::string& err) {
    Mat m;
    TensorRef one = t;
    if (t.shape.size() == 1) {
        one.shape = {t.shape[0], 1};
    } else if (t.shape.size() == 2 && t.shape[1] == 1) {
        // already [n, 1]
    } else {
        err = t.name + ": expected a vector";
        return false;
    }
    if (!read_mat(path, one, m, err)) return false;
    out = std::move(m.w);
    return true;
}

// ------------------------------------------------------------ rope

float rope_inv_freq(const RopeParams& p, int i) {
    float theta = std::pow(p.base, -2.f * i / p.rot);
    if (p.n_ctx_orig > 0 && p.freq_scale != 1.f) {
        const float pi = 3.14159265358979323846f;
        const float wavelen = 2.f * pi / theta;
        const float low = (float) p.n_ctx_orig / p.beta_fast;    // keep below
        const float high = (float) p.n_ctx_orig / p.beta_slow;   // full scale above
        if (wavelen > high) {
            theta *= p.freq_scale;
        } else if (wavelen >= low) {
            const float smooth = ((float) p.n_ctx_orig / wavelen - p.beta_slow) / (p.beta_fast - p.beta_slow);
            theta = theta * (1.f - smooth) + theta * p.freq_scale * smooth;
        }
    }
    return theta;
}

void rope_inplace(float* x, int off, int64_t pos, const RopeParams& p) {
    for (int i = 0; i < p.rot / 2; ++i) {
        const float a = (float) pos * rope_inv_freq(p, i);
        const float c = std::cos(a), s = std::sin(a);
        float& x0 = x[off + 2 * i];
        float& x1 = x[off + 2 * i + 1];
        const float v0 = x0, v1 = x1;
        x0 = v0 * c - v1 * s;
        x1 = v0 * s + v1 * c;
    }
}

void rope_back_inplace(float* x, int off, int64_t pos, const RopeParams& p) {
    for (int i = 0; i < p.rot / 2; ++i) {
        const float a = (float) pos * rope_inv_freq(p, i);
        const float c = std::cos(a), s = std::sin(a);
        float& x0 = x[off + 2 * i];
        float& x1 = x[off + 2 * i + 1];
        const float v0 = x0, v1 = x1;
        x0 = v0 * c + v1 * s;
        x1 = -v0 * s + v1 * c;
    }
}

void rope_neox_inplace(float* x, int64_t pos, const RopeParams& p) {
    for (int i = 0; i < p.rot / 2; ++i) {
        const float a = (float) pos * rope_inv_freq(p, i);
        const float c = std::cos(a), s = std::sin(a);
        float& x0 = x[i];
        float& x1 = x[i + p.rot / 2];
        const float v0 = x0, v1 = x1;
        x0 = v0 * c - v1 * s;
        x1 = v0 * s + v1 * c;
    }
}

// ------------------------------------------------------------ loaders

bool load_hca(const std::string& path, const Pack& p, int layer, HcaWeights& w, std::string& err) {
    const Geometry& g = p.geo;
    if (!is_hca(g, layer)) {
        err = "layer " + std::to_string(layer) + " is not HCA";
        return false;
    }
    const std::string pre = "blk." + std::to_string(layer) + ".";
    const int nh = g.n_head, hd = g.head_dim, ratio = g.compress_ratio[layer];
    const int og = g.o_groups, ol = g.o_lora;
    Mat vha;
    if (!get_mat(path, p, pre + "attn_q_a.weight", w.q_a, err) ||
        !get_mat(path, p, pre + "attn_q_b.weight", w.q_b, err) ||
        !get_mat(path, p, pre + "attn_kv.weight", w.kv, err) ||
        !get_mat(path, p, pre + "attn_output_a.weight", w.o_a, err) ||
        !get_mat(path, p, pre + "attn_output_b.weight", w.o_b, err) ||
        !get_mat(path, p, pre + "attn_gate.weight", w.gate, err) ||
        !get_mat(path, p, pre + "attn_compressor_kv.weight", w.comp_kv, err) ||
        !get_mat(path, p, pre + "attn_compressor_gate.weight", w.comp_gate, err) ||
        !get_mat(path, p, pre + "attn_compressor_ape.weight", w.ape, err) ||
        !get_vec(path, p, pre + "attn_q_a_norm.weight", w.q_a_norm, err) ||
        !get_vec(path, p, pre + "attn_kv_a_norm.weight", w.kv_norm, err) ||
        !get_vec(path, p, pre + "attn_sinks.weight", w.sinks, err) ||
        !get_mat(path, p, pre + "attn_vha.weight", vha, err) ||
        !get_vec(path, p, pre + "attn_compressor_norm.weight", w.comp_norm, err))
        return false;
    w.vha = vha.w;
    return shape_ok(w.q_a, g.q_lora, g.n_embd, pre + "attn_q_a.weight", err) &&
           shape_ok(w.q_b, nh * hd, g.q_lora, pre + "attn_q_b.weight", err) &&
           shape_ok(w.kv, hd, g.n_embd, pre + "attn_kv.weight", err) &&
           shape_ok(w.o_a, og * ol, nh * hd / og, pre + "attn_output_a.weight", err) &&
           shape_ok(w.o_b, g.n_embd, og * ol, pre + "attn_output_b.weight", err) &&
           shape_ok(w.gate, og * ol, g.q_lora, pre + "attn_gate.weight", err) &&
           shape_ok(w.comp_kv, hd, g.n_embd, pre + "attn_compressor_kv.weight", err) &&
           shape_ok(w.comp_gate, hd, g.n_embd, pre + "attn_compressor_gate.weight", err) &&
           shape_ok(w.ape, ratio, hd, pre + "attn_compressor_ape.weight", err) &&
           shape_ok(w.q_a_norm, g.q_lora, pre + "attn_q_a_norm.weight", err) &&
           shape_ok(w.kv_norm, hd, pre + "attn_kv_a_norm.weight", err) &&
           shape_ok(w.sinks, nh, pre + "attn_sinks.weight", err) &&
           shape_ok(vha, nh, nh, pre + "attn_vha.weight", err) &&
           shape_ok(w.comp_norm, hd, pre + "attn_compressor_norm.weight", err);
}

bool load_mla(const std::string& path, const Pack& p, int layer, MlaWeights& w, std::string& err) {
    const Geometry& g = p.geo;
    if (is_hca(g, layer)) {
        err = "layer " + std::to_string(layer) + " is not MLA";
        return false;
    }
    const std::string pre = "blk." + std::to_string(layer) + ".";
    const int nh = g.n_head, nope = g.mla_qk - g.rope_dim, rot = g.rope_dim;
    Mat vha;
    if (!get_mat(path, p, pre + "attn_q_a.weight", w.q_a, err) ||
        !get_mat(path, p, pre + "attn_q_b.weight", w.q_b, err) ||
        !get_mat(path, p, pre + "attn_kv_a_mqa.weight", w.kv_a, err) ||
        !get_mat(path, p, pre + "attn_kv_b.weight", w.kv_b, err) ||
        !get_mat(path, p, pre + "attn_output.weight", w.o, err) ||
        !get_mat(path, p, pre + "attn_gate.weight", w.gate, err) ||
        !get_mat(path, p, pre + "indexer.attn_q_b.weight", w.idx_q_b, err) ||
        !get_mat(path, p, pre + "indexer.attn_k.weight", w.idx_k, err) ||
        !get_mat(path, p, pre + "indexer.proj.weight", w.idx_proj, err) ||
        !get_vec(path, p, pre + "attn_q_a_norm.weight", w.q_a_norm, err) ||
        !get_vec(path, p, pre + "attn_kv_a_norm.weight", w.kv_a_norm, err) ||
        !get_vec(path, p, pre + "attn_sinks.weight", w.sinks, err) ||
        !get_mat(path, p, pre + "attn_vha.weight", vha, err) ||
        !get_vec(path, p, pre + "indexer.k_norm.weight", w.idx_k_norm, err) ||
        !get_vec(path, p, pre + "indexer.k_norm.bias", w.idx_k_norm_b, err))
        return false;
    w.vha = vha.w;
    return shape_ok(w.q_a, g.q_lora, g.n_embd, pre + "attn_q_a.weight", err) &&
           shape_ok(w.q_b, nh * g.mla_qk, g.q_lora, pre + "attn_q_b.weight", err) &&
           shape_ok(w.kv_a, g.kv_lora + rot, g.n_embd, pre + "attn_kv_a_mqa.weight", err) &&
           shape_ok(w.kv_b, nh * (nope + g.mla_v), g.kv_lora, pre + "attn_kv_b.weight", err) &&
           shape_ok(w.o, g.n_embd, nh * g.mla_v, pre + "attn_output.weight", err) &&
           shape_ok(w.gate, nh * g.mla_v, g.q_lora, pre + "attn_gate.weight", err) &&
           shape_ok(w.idx_q_b, g.idx_heads * g.idx_dim, g.q_lora, pre + "indexer.attn_q_b.weight", err) &&
           shape_ok(w.idx_k, g.idx_dim, g.n_embd, pre + "indexer.attn_k.weight", err) &&
           shape_ok(w.idx_proj, g.idx_heads, g.n_embd, pre + "indexer.proj.weight", err) &&
           shape_ok(w.q_a_norm, g.q_lora, pre + "attn_q_a_norm.weight", err) &&
           shape_ok(w.kv_a_norm, g.kv_lora, pre + "attn_kv_a_norm.weight", err) &&
           shape_ok(w.sinks, nh, pre + "attn_sinks.weight", err) &&
           shape_ok(vha, nh, nh, pre + "attn_vha.weight", err) &&
           shape_ok(w.idx_k_norm, g.idx_dim, pre + "indexer.k_norm.weight", err) &&
           shape_ok(w.idx_k_norm_b, g.idx_dim, pre + "indexer.k_norm.bias", err);
}

// ------------------------------------------------------------ HCA

HcaPrepared hca_prepare(const Geometry& g, const HcaWeights& w, const std::vector<Vec>& x) {
    const int T = (int) x.size();
    const int nh = g.n_head, hd = g.head_dim, ratio = g.compress_ratio[0] ? g.compress_ratio[0] : 128;
    const float eps = g.rms_eps;
    const RopeParams rope = hca_rope(g);
    const int off = hd - g.rope_dim;

    HcaPrepared p;
    p.T = T;
    p.qr.resize(T);
    p.q.resize((size_t) T * nh * hd);
    p.kv.resize((size_t) T * hd);
    std::vector<Vec> skv(T), ssc(T);
    for (int t = 0; t < T; ++t) {
        p.qr[t] = matvec(w.q_a, x[t].data());
        rmsnorm_inplace(p.qr[t], &w.q_a_norm, eps);
        Vec qf = matvec(w.q_b, p.qr[t].data());
        for (int h = 0; h < nh; ++h) {
            float* qh = p.q.data() + ((size_t) t * nh + h) * hd;
            std::copy(qf.begin() + h * hd, qf.begin() + (h + 1) * hd, qh);
            Vec qhv(qh, qh + hd);
            rmsnorm_inplace(qhv, nullptr, eps);
            std::copy(qhv.begin(), qhv.end(), qh);
            rope_inplace(qh, off, t, rope);
        }
        float* kvt = p.kv.data() + (size_t) t * hd;
        Vec kvv = matvec(w.kv, x[t].data());
        rmsnorm_inplace(kvv, &w.kv_norm, eps);
        std::copy(kvv.begin(), kvv.end(), kvt);
        rope_inplace(kvt, off, t, rope);
        skv[t] = matvec(w.comp_kv, x[t].data());
        ssc[t] = matvec(w.comp_gate, x[t].data());
        for (int d = 0; d < hd; ++d) ssc[t][d] += w.ape.row(t % ratio)[d];
    }

    // the compressor: one entry per completed block, a per-dim softmax over the block's positions
    const int nb = T / ratio;
    p.nb = nb;
    p.comp.resize((size_t) nb * hd, 0.f);
    for (int b = 0; b < nb; ++b) {
        Vec compb(hd, 0.f);
        for (int d = 0; d < hd; ++d) {
            float mx = -kInf;
            for (int i = 0; i < ratio; ++i) mx = std::max(mx, ssc[b * ratio + i][d]);
            double den = 0.0;
            for (int i = 0; i < ratio; ++i) den += std::exp((double) ssc[b * ratio + i][d] - mx);
            for (int i = 0; i < ratio; ++i)
                compb[d] += (float) (std::exp((double) ssc[b * ratio + i][d] - mx) / den) *
                            skv[b * ratio + i][d];
        }
        rmsnorm_inplace(compb, &w.comp_norm, eps);
        std::copy(compb.begin(), compb.end(), p.comp.data() + (size_t) b * hd);
        rope_inplace(p.comp.data() + (size_t) b * hd, off, (int64_t) (b + 1) * ratio - 1, rope);
    }
    return p;
}

void hca_attention_ref(const Geometry& g, const HcaWeights& w, const HcaPrepared& p, int t, int h,
                       float* out) {
    const int nh = g.n_head, hd = g.head_dim, ratio = g.compress_ratio[0] ? g.compress_ratio[0] : 128;
    const float scale = 1.f / std::sqrt((float) hd);
    const int j0 = std::max(0, t - g.window + 1);
    const int nraw = t - j0 + 1;
    const int ncomp = t / ratio;   // completed blocks strictly before this token's block
    const float* qh = p.q.data() + ((size_t) t * nh + h) * hd;
    Vec scores(nraw + ncomp);
    for (int j = j0; j <= t; ++j)
        scores[j - j0] = scale * dot(qh, p.kv.data() + (size_t) j * hd, hd);
    for (int b = 0; b < ncomp; ++b)
        scores[nraw + b] = scale * dot(qh, p.comp.data() + (size_t) b * hd, hd);
    const Vec pr = softmax_with_sink(scores, w.sinks[h]);
    std::fill(out, out + hd, 0.f);
    for (int j = j0; j <= t; ++j)
        for (int d = 0; d < hd; ++d) out[d] += pr[j - j0] * p.kv[(size_t) j * hd + d];
    for (int b = 0; b < ncomp; ++b)
        for (int d = 0; d < hd; ++d) out[d] += pr[nraw + b] * p.comp[(size_t) b * hd + d];
}

std::vector<Vec> hca_forward(const Geometry& g, const HcaWeights& w, const std::vector<Vec>& x) {
    const HcaPrepared p = hca_prepare(g, w, x);
    const int T = p.T;
    const int nh = g.n_head, hd = g.head_dim;
    const int og = g.o_groups, ol = g.o_lora, hpg = nh / og;
    const RopeParams rope = hca_rope(g);
    const int off = hd - g.rope_dim;

    std::vector<Vec> out(T, Vec(g.n_embd, 0.f));
    for (int t = 0; t < T; ++t) {
        std::vector<Vec> heads(nh, Vec(hd, 0.f));
        for (int h = 0; h < nh; ++h) {
            hca_attention_ref(g, w, p, t, h, heads[h].data());
            rope_back_inplace(heads[h].data(), off, t, rope);
        }
        head_mix(heads, w.vha);
        const Vec gatev = matvec(w.gate, p.qr[t].data());
        Vec oa(og * ol);
        for (int gg = 0; gg < og; ++gg) {
            Vec slice(hpg * hd);
            for (int hh = 0; hh < hpg; ++hh)
                std::copy(heads[gg * hpg + hh].begin(), heads[gg * hpg + hh].end(),
                          slice.begin() + hh * hd);
            for (int r = 0; r < ol; ++r) {
                const float acc = dot(w.o_a.row(gg * ol + r), slice.data(), hpg * hd);
                oa[gg * ol + r] = acc * sigmoid(gatev[gg * ol + r]);
            }
        }
        out[t] = matvec(w.o_b, oa.data());
    }
    return out;
}

// ------------------------------------------------------------ MLA

MlaResult mla_forward(const Geometry& g, const MlaWeights& w, const std::vector<Vec>& x) {
    const int T = (int) x.size();
    const int nh = g.n_head, nope = g.mla_qk - g.rope_dim, rot = g.rope_dim, v = g.mla_v;
    const float eps = g.rms_eps;
    const RopeParams rope = plain_rope(g);
    const float scale = 1.f / std::sqrt((float) g.mla_qk);

    std::vector<Vec> q_c(T), gate(T), kv_c(T), k_pe(T);
    std::vector<std::vector<Vec>> q_abs(T), q_pe(T);
    for (int t = 0; t < T; ++t) {
        q_c[t] = matvec(w.q_a, x[t].data());
        rmsnorm_inplace(q_c[t], &w.q_a_norm, eps);
        const Vec gv = matvec(w.gate, q_c[t].data());
        gate[t].resize(nh * v);
        for (int i = 0; i < nh * v; ++i) gate[t][i] = sigmoid(gv[i]);
        const Vec qf = matvec(w.q_b, q_c[t].data());
        const Vec kvf = matvec(w.kv_a, x[t].data());
        kv_c[t].assign(kvf.begin(), kvf.begin() + g.kv_lora);
        rmsnorm_inplace(kv_c[t], &w.kv_a_norm, eps);
        k_pe[t].assign(kvf.begin() + g.kv_lora, kvf.end());
        rope_inplace(k_pe[t].data(), 0, t, rope);
        q_abs[t].resize(nh);
        q_pe[t].resize(nh);
        for (int h = 0; h < nh; ++h) {
            const float* qh = qf.data() + h * g.mla_qk;
            // absorb k_b into the query: rows [h*(nope+v), h*(nope+v)+nope) of kv_b
            q_abs[t][h].assign(g.kv_lora, 0.f);
            for (int c = 0; c < g.kv_lora; ++c) {
                double acc = 0.0;
                for (int n = 0; n < nope; ++n) acc += (double) qh[n] * w.kv_b.row(h * (nope + v) + n)[c];
                q_abs[t][h][c] = (float) acc;
            }
            q_pe[t][h].assign(qh + nope, qh + nope + rot);
            rope_inplace(q_pe[t][h].data(), 0, t, rope);
        }
    }

    MlaResult res;
    res.out.assign(T, Vec(g.n_embd, 0.f));
    for (int t = 0; t < T; ++t) {
        std::vector<Vec> heads(nh, Vec(v, 0.f));
        Vec scores(t + 1);
        for (int h = 0; h < nh; ++h) {
            Vec Q(g.kv_lora + rot);
            std::copy(q_abs[t][h].begin(), q_abs[t][h].end(), Q.begin());
            std::copy(q_pe[t][h].begin(), q_pe[t][h].end(), Q.begin() + g.kv_lora);
            for (int j = 0; j <= t; ++j) {
                const float s1 = dot(Q.data(), kv_c[j].data(), g.kv_lora);
                const float s2 = dot(Q.data() + g.kv_lora, k_pe[j].data(), rot);
                scores[j] = scale * (s1 + s2);
            }
            const Vec p = softmax_with_sink(scores, w.sinks[h]);
            Vec a(g.kv_lora, 0.f);
            for (int j = 0; j <= t; ++j)
                for (int c = 0; c < g.kv_lora; ++c) a[c] += p[j] * kv_c[j][c];            for (int r = 0; r < v; ++r)   // v_b: rows [h*(nope+v)+nope, (h+1)*(nope+v))
                heads[h][r] = dot(w.kv_b.row(h * (nope + v) + nope + r), a.data(), g.kv_lora);
        }
        head_mix(heads, w.vha);
        Vec mix(nh * v);
        for (int h = 0; h < nh; ++h)
            for (int r = 0; r < v; ++r) mix[h * v + r] = heads[h][r] * gate[t][h * v + r];
        res.out[t] = matvec(w.o, mix.data());
    }

    // the lightning indexer: LayerNorm + NEOX rope keys, per-head relu scores weighted by a projection
    res.idx_scores.assign(T, Vec(T, -kInf));
    std::vector<Vec> ik(T), iw(T);
    std::vector<std::vector<Vec>> iq(T);
    for (int t = 0; t < T; ++t) {
        ik[t] = matvec(w.idx_k, x[t].data());
        layernorm_inplace(ik[t], w.idx_k_norm, w.idx_k_norm_b, 1e-6f);
        rope_neox_inplace(ik[t].data(), t, rope);
        const Vec iqf = matvec(w.idx_q_b, q_c[t].data());
        iq[t].resize(g.idx_heads);
        for (int h = 0; h < g.idx_heads; ++h) {
            iq[t][h].assign(iqf.begin() + h * g.idx_dim, iqf.begin() + (h + 1) * g.idx_dim);
            rope_neox_inplace(iq[t][h].data(), t, rope);
        }
        iw[t] = matvec(w.idx_proj, x[t].data());
        const float is = 1.f / std::sqrt((float) (g.idx_dim * g.idx_heads));
        for (float& f : iw[t]) f *= is;
    }
    for (int t = 0; t < T; ++t)
        for (int j = 0; j <= t; ++j) {
            float s = 0.f;
            for (int h = 0; h < g.idx_heads; ++h)
                s += std::max(0.f, dot(iq[t][h].data(), ik[j].data(), g.idx_dim)) * iw[t][h];
            res.idx_scores[t][j] = s;
        }
    return res;
}

}  // namespace basalt::models::ernie55::ref
