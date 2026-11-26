// src/models/ernie55/pack.cpp - the ERNIE 5.5 Flash Preview pack's structural loader: opens a converted
// .basalt and enforces the iteration-3 contract (metadata geometry, exact tensor table, sections, experts
// layout, drafter manifest). No device work; the compute path loads through this and still refuses to decode.
#include "models/ernie55/pack.hpp"

#include <string>

namespace basalt::models::ernie55 {
namespace {

using core::Json;

/// The pack's `ernie55.*` metadata (meta.gguf_kv) as the pure geometry reader's Kv.
Kv kv_of(const Json& gj) {
    Kv kv;
    kv.u = [&](const std::string& k, uint64_t& out) {
        const Json* v = gj.get(k);
        if (!v || !v->is_number()) return false;
        out = v->u64();
        return true;
    };
    kv.f = [&](const std::string& k, double& out) {
        const Json* v = gj.get(k);
        if (!v || !v->is_number()) return false;
        out = v->num();
        return true;
    };
    kv.u_arr = [&](const std::string& k, std::vector<uint64_t>& out) {
        out.clear();
        const Json* v = gj.get(k);
        if (!v || !v->is_array()) return false;
        if (v->count != v->items.size()) return false;   // a truncated array is a lie; the contract needs it whole
        for (const Json& x : v->items) out.push_back(x.u64());
        return true;
    };
    kv.f_arr = [&](const std::string& k, std::vector<double>& out) {
        out.clear();
        const Json* v = gj.get(k);
        if (!v || !v->is_array()) return false;
        if (v->count != v->items.size()) return false;
        for (const Json& x : v->items) out.push_back(x.num());
        return true;
    };
    return kv;
}

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "ernie55") {
        err = path + ": not an ernie55 pack";
        return false;
    }
    const Json* gj = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!gj || !gj->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    if (err = geometry_from(kv_of(*gj), p.geo); !err.empty()) {
        err = path + ": " + err;
        return false;
    }
    if (err = check_geometry(p.geo); !err.empty()) {
        err = path + ": " + err;
        return false;
    }

    // every section the contract names
    for (const char* s : required_sections())
        if (!p.file.find(s)) {
            err = path + ": missing section " + s;
            return false;
        }

    // the tensor table: exactly the contract, no missing, no extra, matching types and bytes
    std::map<std::string, TensorRef> got;
    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        const Json* ty = t.get("type_id");
        const Json* sh = t.get("shape");
        const Json* by = t.get("bytes");
        const Json* off = t.get("offset");
        if (!ty || !sh || !by || !off || !sh->is_array()) {
            err = path + ": meta.tensors." + name + " is incomplete";
            return false;
        }
        r.type = (int) ty->u64();
        for (const Json& d : sh->items) r.shape.push_back(d.u64());
        r.bytes = by->u64();
        std::string section = "gguf";
        if (const Json* s = t.get("section"); s && s->is_string()) section = s->s;
        const core::ModelFile::Section* sec = p.file.find(section);
        if (!sec) {
            err = path + ": tensor " + name + " names a missing section " + section;
            return false;
        }
        r.offset = sec->offset + off->u64();
        if (off->u64() + r.bytes > sec->bytes) {
            err = path + ": tensor " + name + " runs past its section";
            return false;
        }
        uint64_t values = 1;
        for (uint64_t d : r.shape) values *= d;
        if (tensor_bytes(r.type, values) != r.bytes) {
            err = path + ": tensor " + name + " bytes do not match its type and shape";
            return false;
        }
        got.emplace(name, std::move(r));
    }
    const std::vector<TensorSpec> want = expected_tensors(p.geo);
    if (got.size() != want.size()) {
        err = path + ": meta.tensors has " + std::to_string(got.size()) + " tensors, the contract wants " +
              std::to_string(want.size());
        return false;
    }
    for (const TensorSpec& w : want) {
        auto it = got.find(w.name);
        if (it == got.end()) {
            err = path + ": missing contract tensor " + w.name;
            return false;
        }
        if (it->second.type != w.type) {
            err = path + ": tensor " + w.name + " is type " + std::to_string(it->second.type) + ", the contract "
                  "wants " + std::to_string(w.type);
            return false;
        }
    }
    p.tensors = std::move(got);

    // the experts section and its layout
    p.experts_range = p.file.range(kSectionExperts);
    std::vector<uint8_t> text;
    if (!p.file.read(kSectionExpertsLayout, text, err)) return false;
    if (err = parse_experts_layout(std::string(text.begin(), text.end()), p.geo, p.experts); !err.empty()) {
        err = path + ": " + err;
        return false;
    }
    const uint64_t packed = expert_blob_bytes(p.geo) * (uint64_t) p.geo.n_expert * p.experts.size();
    if (packed != p.experts_range.size()) {
        err = path + ": experts section is " + std::to_string(p.experts_range.size()) + " bytes, the layout and "
              "geometry imply " + std::to_string(packed);
        return false;
    }

    // the raw drafter: the manifest's bytes are exactly the section's
    p.dspark_raw = p.file.range(kSectionDsparkRaw);
    std::vector<uint8_t> mj;
    if (!p.file.read(kSectionDsparkManifest, mj, err)) return false;
    if (!core::parse_json((const char*) mj.data(), mj.size(), p.dspark_manifest, err)) {
        err = path + ": dspark/manifest.json: " + err;
        return false;
    }
    if (!p.dspark_manifest.is_object() || p.dspark_manifest.members.empty()) {
        err = path + ": dspark/manifest.json is empty";
        return false;
    }
    uint64_t total = 0;
    for (const auto& [name, e] : p.dspark_manifest.members) {
        const Json* off = e.get("offset");
        const Json* by = e.get("bytes");
        if (!off || !by || !e.get("dtype") || !e.get("shape")) {
            err = path + ": dspark manifest entry " + name + " is incomplete";
            return false;
        }
        if (off->u64() != total) {
            err = path + ": dspark manifest entry " + name + " is out of the packed order";
            return false;
        }
        total += by->u64();
    }
    if (total != p.dspark_raw.size()) {
        err = path + ": dspark/raw is " + std::to_string(p.dspark_raw.size()) + " bytes, the manifest implies " +
              std::to_string(total);
        return false;
    }
    return true;
}

}  // namespace basalt::models::ernie55
