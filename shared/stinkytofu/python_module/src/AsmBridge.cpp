// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

namespace nb = nanobind;
using namespace stinkytofu;

namespace {

/// The AsmBuilder method running on this thread, or null. The checks AsmBuilder shares with
/// emit_asm call fail() with an emit_asm record index; while a method runs, they report under
/// its name instead.
thread_local const char* tAsmBuilderMethod = nullptr;

[[noreturn]] void fail(size_t itemIndex, const std::string& message) {
    if (tAsmBuilderMethod != nullptr)
        throw nb::value_error(
            ("AsmBuilder." + std::string(tAsmBuilderMethod) + ": " + message).c_str());
    throw nb::value_error(("emit_asm item[" + std::to_string(itemIndex) + "]: " + message).c_str());
}

nb::handle find(const nb::dict& record, const char* key) {
    PyObject* value = PyDict_GetItemString(record.ptr(), key);
    return value ? nb::handle(value) : nb::handle();
}

nb::handle require(const nb::dict& record, const char* key, size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) fail(itemIndex, std::string("missing required field '") + key + "'");
    return value;
}

std::string requireString(const nb::dict& record, const char* key, size_t itemIndex) {
    nb::handle value = require(record, key, itemIndex);
    if (!PyUnicode_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be a string");
    return nb::cast<std::string>(value);
}

int64_t requireInteger(nb::handle value, const std::string& field, size_t itemIndex) {
    if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
        fail(itemIndex, "field '" + field + "' must be an integer");
    try {
        return nb::cast<int64_t>(value);
    } catch (const nb::cast_error&) {
        fail(itemIndex, "field '" + field + "' is outside the signed 64-bit range");
    }
}

bool optionalBool(const nb::dict& record, const char* key, bool defaultValue, size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) return defaultValue;
    if (!PyBool_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be bool");
    return value.ptr() == Py_True;
}

int64_t optionalInteger(const nb::dict& record, const char* key, int64_t defaultValue,
                        size_t itemIndex) {
    nb::handle value = find(record, key);
    return value.is_valid() ? requireInteger(value, key, itemIndex) : defaultValue;
}

std::vector<int> optionalIntegerVector(const nb::dict& record, const char* key,
                                       size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) return {};
    if (!PyList_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be a list of integers");
    std::vector<int> result;
    for (nb::handle element : nb::borrow<nb::list>(value)) {
        int64_t parsed = requireInteger(element, key, itemIndex);
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())
            fail(itemIndex, std::string("field '") + key + "' value is outside the int range");
        result.push_back(static_cast<int>(parsed));
    }
    return result;
}

void rejectUnknownKeys(const nb::dict& record, const std::unordered_set<std::string>& allowed,
                       size_t itemIndex) {
    Py_ssize_t pos = 0;
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    while (PyDict_Next(record.ptr(), &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) fail(itemIndex, "record keys must be strings");
        std::string name = nb::cast<std::string>(nb::handle(key));
        if (!allowed.contains(name)) fail(itemIndex, "unknown field '" + name + "'");
    }
}

bool isLabelName(const std::string& name) {
    if (name.empty()) return false;
    auto first = static_cast<unsigned char>(name.front());
    if (!(std::isalpha(first) || name.front() == '_' || name.front() == '.' ||
          name.front() == '$'))
        return false;
    return std::all_of(name.begin() + 1, name.end(), [](char c) {
        auto u = static_cast<unsigned char>(c);
        return std::isalnum(u) || c == '_' || c == '.' || c == '$';
    });
}

bool isSupportedTypedInstruction(const std::string& mnemonic) {
    static const std::unordered_set<std::string> supported = {
        // T00 FMHA typed forms. Packed v_pk_fma_f32/v_pk_mul_f32 remain audited custom words.
        // buffer_atomic_add_f32 is accepted only in its ::noret form (see selectForm).
        "buffer_atomic_add_f32",
        "ds_load_b128", "ds_load_tr16_b128", "ds_store_b128", "ds_store_b32",
        "flat_load_b32", "flat_store_b32", "global_prefetch_b8", "s_add_co_ci_u32",
        "s_add_co_i32", "s_add_co_u32", "s_add_u32",
        "s_addk_co_i32", "s_and_b32", "s_ashr_i32", "s_barrier_signal",
        "s_barrier_wait", "s_bfe_u32", "s_bitcmp1_b32", "s_bitset0_b32",
        "s_bitset1_b32", "s_cselect_b32", "s_cmp_eq_u32", "s_cmp_ge_u32",
        "s_cmp_lt_i32", "s_cmp_lt_u32", "s_code_end", "s_endpgm", "s_getpc_b64",
        "s_lshl2_add_u32", "s_lshl_b32", "s_lshr_b32", "s_load_b32", "s_load_b64",
        "s_max_i32", "s_min_i32", "s_min_u32", "s_mov_b32", "s_mul_hi_u32",
        "s_mul_i32", "s_nop", "s_or_b32", "s_prefetch_inst", "s_setreg_IMM32_b32",
        "s_sub_co_i32", "s_sub_u32", "s_version", "s_wait_alu", "s_wait_dscnt",
        "s_wait_idle", "s_wait_kmcnt", "s_wait_loadcnt", "s_wait_storecnt",
        "s_wait_tensorcnt", "s_waitcnt", "s_xor_b32",
        "tensor_load_to_lds", "tensor_store_from_lds", "v_add_co_ci_u32", "v_add_co_u32",
        "v_add_f32", "v_add_nc_i32", "v_add_nc_u32", "v_add_nc_u64", "v_and_b32",
        "v_cmp_eq_f32", "v_cmp_le_u32", "v_cmp_lt_i32",
        "v_cmp_lt_u32", "v_cndmask_b32", "v_cvt_f32_u32", "v_cvt_pk_bf16_f32",
        "v_cvt_u32_f32", "v_exp_f32", "v_fma_f32", "v_log_f32", "v_lshlrev_b32",
        "v_lshrrev_b32", "v_mad_co_u64_u32", "v_max3_num_f32", "v_max_num_f32", "v_mov_b32",
        "v_mov_b64",
        "v_mul_f32", "v_mul_hi_u32", "v_mul_i32_i24", "v_mul_lo_u32",
        "v_mul_u32_u24", "v_nop", "v_permlane16_swap_b32", "v_permlanex16_b32",
        "v_perm_b32", "v_pk_add_f32", "v_pk_mul_f32", "v_rcp_f32", "v_rcp_iflag_f32", "v_readfirstlane_b32",
        "v_readlane_b32", "v_sub_f32", "v_sub_nc_i32", "v_sub_u32", "v_subrev_nc_u32",
        "v_swap_b32", "v_tanh_f32", "v_wmma_f32_16x16x32_bf16", "v_writelane_b32", "v_xor3_b32",
        "v_xor_b32",
        // Structural records selected by the target lowering.
        "s_branch", "s_cbranch_execnz", "s_cbranch_execz", "s_cbranch_scc0",
        "s_cbranch_scc1", "s_cbranch_vccnz", "s_cbranch_vccz", "s_set_vgpr_msb",
    };
    return supported.contains(mnemonic);
}

std::vector<StinkyRegister> parseOperands(const nb::dict& record, const char* key,
                                          size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) return {};
    if (!PyList_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be a list of Register values");

    std::vector<StinkyRegister> result;
    nb::list values = nb::borrow<nb::list>(value);
    result.reserve(values.size());
    size_t operandIndex = 0;
    for (nb::handle operand : values) {
        if (!nb::isinstance<StinkyRegister>(operand)) {
            fail(itemIndex, std::string("field '") + key + "'[" +
                                std::to_string(operandIndex) + "] must be a Register");
        }
        result.push_back(nb::cast<StinkyRegister>(operand));
        ++operandIndex;
    }
    return result;
}

// gfx1250 buffer instructions encode an unsigned 23-bit immediate offset.
constexpr int64_t kMaxMubufOffset = (int64_t{1} << 23) - 1;

/// Apply a record's 'modifiers' value to \p inst; \p value is invalid when the record has none.
void addModifiers(StinkyInstruction& inst, nb::handle value, size_t itemIndex) {
    // The typed buffer atomic always carries a VGPR vaddr, so it needs the offen addressing
    // mode; without the modifier the emitter would print a line that does not assemble.
    const bool requiresMubuf =
        std::string_view(inst.getHwInstDesc()->mnemonic) == "buffer_atomic_add_f32";
    if (!value.is_valid()) {
        if (requiresMubuf)
            fail(itemIndex, "buffer_atomic_add_f32 requires modifier 'mubuf' with offen=True");
        return;
    }
    if (!PyDict_Check(value.ptr())) fail(itemIndex, "field 'modifiers' must be a dict");
    nb::dict modifiers = nb::borrow<nb::dict>(value);
    rejectUnknownKeys(modifiers,
                      {"ds", "global", "mubuf", "smem", "vop3p", "mfma", "swaitcnt",
                       "swaittensorcnt", "waitalu"},
                      itemIndex);
    if (requiresMubuf && !find(modifiers, "mubuf").is_valid())
        fail(itemIndex, "buffer_atomic_add_f32 requires modifier 'mubuf' with offen=True");

    if (nb::handle dsValue = find(modifiers, "ds"); dsValue.is_valid()) {
        if (!(inst.is(IF_DSRead) || inst.is(IF_DSStore) || inst.is(IF_DSAtomic)))
            fail(itemIndex, "modifier 'ds' is only valid on DS instructions");
        if (!PyDict_Check(dsValue.ptr())) fail(itemIndex, "modifier 'ds' must be a dict");
        nb::dict ds = nb::borrow<nb::dict>(dsValue);
        rejectUnknownKeys(ds, {"na", "offset", "offset0", "offset1", "gds"}, itemIndex);

        int64_t na = optionalInteger(ds, "na", 1, itemIndex);
        int64_t offset = optionalInteger(ds, "offset", 0, itemIndex);
        int64_t offset0 = optionalInteger(ds, "offset0", 0, itemIndex);
        int64_t offset1 = optionalInteger(ds, "offset1", 0, itemIndex);
        if (na < 1 || na > 2) fail(itemIndex, "DS modifier 'na' must be 1 or 2");
        if (offset < 0 || offset > 65535)
            fail(itemIndex, "DS modifier 'offset' must be in [0, 65535]");
        if (offset0 < 0 || offset0 > 255 || offset1 < 0 || offset1 > 255)
            fail(itemIndex, "DS modifier offset0/offset1 must be in [0, 255]");
        if (na == 1 && (offset0 != 0 || offset1 != 0))
            fail(itemIndex, "DS modifier offset0/offset1 require na=2");
        if (optionalBool(ds, "gds", false, itemIndex))
            fail(itemIndex, "DS modifier gds=True is unsupported on the gfx1250 bridge");
        inst.addModifier(DSModifiers(static_cast<int>(na), static_cast<int>(offset),
                                     static_cast<int>(offset0), static_cast<int>(offset1),
                                     false));
    }

    if (nb::handle value = find(modifiers, "vop3p"); value.is_valid()) {
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'vop3p' must be a dict");
        if (inst.getHwInstDesc()->microcode != MicrocodeFormat::MC_VOP3P)
            fail(itemIndex, "modifier 'vop3p' is only valid on VOP3P instructions");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"op_sel", "op_sel_hi", "byte_sel"}, itemIndex);
        auto opSel = optionalIntegerVector(data, "op_sel", itemIndex);
        auto opSelHi = optionalIntegerVector(data, "op_sel_hi", itemIndex);
        auto byteSel = optionalIntegerVector(data, "byte_sel", itemIndex);
        for (const auto* pair : {&opSel, &opSelHi, &byteSel})
            if (std::any_of(pair->begin(), pair->end(), [](int bit) { return bit != 0 && bit != 1; }))
                fail(itemIndex, "VOP3P selector fields must contain only 0 or 1");
        inst.addModifier(VOP3PModifiers(opSel, opSelHi, byteSel));
    }

    if (nb::handle value = find(modifiers, "mfma"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) !=
            "v_wmma_f32_16x16x32_bf16")
            fail(itemIndex,
                 "modifier 'mfma' is only valid on v_wmma_f32_16x16x32_bf16");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'mfma' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"neg", "reuse_a", "reuse_b"}, itemIndex);
        MFMAModifiers modifier;
        if (optionalBool(data, "neg", false, itemIndex)) {
            modifier.negBits.negLo = {1, 1, 0};
            modifier.negBits.numSrcs = 2;
        }
        modifier.reuseA = optionalBool(data, "reuse_a", false, itemIndex);
        modifier.reuseB = optionalBool(data, "reuse_b", false, itemIndex);
        inst.addModifier(modifier);
    }

    if (nb::handle value = find(modifiers, "smem"); value.is_valid()) {
        const std::string_view mnemonic(inst.getHwInstDesc()->mnemonic);
        if (mnemonic != "s_load_b32" && mnemonic != "s_load_b64")
            fail(itemIndex, "modifier 'smem' is only valid on s_load_b32/s_load_b64");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'smem' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"glc", "nv", "offset"}, itemIndex);
        int64_t offset = optionalInteger(data, "offset", 0, itemIndex);
        if (offset < std::numeric_limits<int>::min() || offset > std::numeric_limits<int>::max())
            fail(itemIndex, "SMEM modifier 'offset' is outside the int range");
        inst.addModifier(SMEMModifiers(optionalBool(data, "glc", false, itemIndex),
                                       optionalBool(data, "nv", false, itemIndex),
                                       static_cast<int>(offset)));
    }

    if (nb::handle value = find(modifiers, "mubuf"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "buffer_atomic_add_f32")
            fail(itemIndex, "modifier 'mubuf' is only valid on buffer_atomic_add_f32");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'mubuf' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"offen", "offset", "scope"}, itemIndex);
        // The emitter prints the offset together with offen, and the typed form always
        // carries a VGPR vaddr, so the offen (VGPR byte offset) addressing mode is required.
        if (!optionalBool(data, "offen", false, itemIndex))
            fail(itemIndex, "MUBUF modifier 'offen' must be True");
        int64_t offset = optionalInteger(data, "offset", 0, itemIndex);
        if (offset < 0 || offset > kMaxMubufOffset)
            fail(itemIndex, "MUBUF modifier 'offset' must be in [0, 8388607]");
        std::string scope = find(data, "scope").is_valid()
                                ? requireString(data, "scope", itemIndex)
                                : "SCOPE_NONE";
        MUBUFScope parsedScope = parseMUBUFScope(scope);
        if (scope != "SCOPE_NONE" && parsedScope == MUBUFScope::SCOPE_NONE)
            fail(itemIndex, "unknown MUBUF scope '" + scope + "'");
        inst.addModifier(MUBUFModifiers(/*offen=*/true, static_cast<int>(offset), /*glc=*/false,
                                        /*slc=*/false, /*nt=*/false, /*lds=*/false,
                                        /*isStore=*/false, /*hasGLCModifier=*/false,
                                        /*hasSC0Modifier=*/false, parsedScope));
    }

    if (nb::handle value = find(modifiers, "global"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "global_prefetch_b8")
            fail(itemIndex, "modifier 'global' is only valid on global_prefetch_b8");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'global' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"offset", "scope", "temporal_hint"}, itemIndex);
        int64_t offset = optionalInteger(data, "offset", 0, itemIndex);
        if (offset < 0 || offset > std::numeric_limits<int>::max())
            fail(itemIndex, "GLOBAL modifier 'offset' must be a nonnegative int");
        std::string scope = find(data, "scope").is_valid()
                                ? requireString(data, "scope", itemIndex)
                                : "SCOPE_NONE";
        std::string hint = find(data, "temporal_hint").is_valid()
                               ? requireString(data, "temporal_hint", itemIndex)
                               : "TH_NONE";
        MUBUFScope parsedScope = parseMUBUFScope(scope);
        TemporalHint parsedHint = parseTemporalHint(hint);
        if (scope != "SCOPE_NONE" && parsedScope == MUBUFScope::SCOPE_NONE)
            fail(itemIndex, "unknown GLOBAL scope '" + scope + "'");
        if (hint != "TH_NONE" && parsedHint == TemporalHint::TH_NONE)
            fail(itemIndex, "unknown GLOBAL temporal_hint '" + hint + "'");
        inst.addModifier(GLOBALModifiers(static_cast<int>(offset), parsedHint, parsedScope));
    }

    if (nb::handle value = find(modifiers, "swaitcnt"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "s_waitcnt")
            fail(itemIndex, "modifier 'swaitcnt' is only valid on s_waitcnt");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'swaitcnt' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"vlcnt", "vscnt", "dscnt", "kmcnt", "wait_all"}, itemIndex);
        if (optionalBool(data, "wait_all", false, itemIndex))
            inst.addModifier(SWaitCntData(0, 0, -1, 0, 0));
        else
            inst.addModifier(SWaitCntData(optionalInteger(data, "vlcnt", -1, itemIndex),
                                          optionalInteger(data, "vscnt", -1, itemIndex), -1,
                                          optionalInteger(data, "dscnt", -1, itemIndex),
                                          optionalInteger(data, "kmcnt", -1, itemIndex)));
    }

    if (nb::handle value = find(modifiers, "swaittensorcnt"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "s_wait_tensorcnt")
            fail(itemIndex,
                 "modifier 'swaittensorcnt' is only valid on s_wait_tensorcnt");
        if (!PyDict_Check(value.ptr()))
            fail(itemIndex, "modifier 'swaittensorcnt' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"value"}, itemIndex);
        int64_t count = optionalInteger(data, "value", -1, itemIndex);
        if (count < -1 || count > 127)
            fail(itemIndex, "S_WAIT_TENSORCNT value must be in [-1, 127]");
        inst.addModifier(SWaitTensorCntData(static_cast<int8_t>(count)));
    }

    if (nb::handle value = find(modifiers, "waitalu"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "s_wait_alu")
            fail(itemIndex, "modifier 'waitalu' is only valid on s_wait_alu");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'waitalu' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data,
                          {"va_vdst", "va_sdst", "va_ssrc", "hold_cnt", "vm_vsrc",
                           "va_vcc", "sa_sdst"},
                          itemIndex);
        int64_t vaVdst = optionalInteger(data, "va_vdst", -1, itemIndex);
        int64_t vaSdst = optionalInteger(data, "va_sdst", -1, itemIndex);
        int64_t vaSsrc = optionalInteger(data, "va_ssrc", -1, itemIndex);
        int64_t holdCnt = optionalInteger(data, "hold_cnt", -1, itemIndex);
        int64_t vmVsrc = optionalInteger(data, "vm_vsrc", -1, itemIndex);
        int64_t vaVcc = optionalInteger(data, "va_vcc", -1, itemIndex);
        int64_t saSdst = optionalInteger(data, "sa_sdst", -1, itemIndex);
        auto inRange = [](int64_t value, int64_t maximum) {
            return value == -1 || (value >= 0 && value <= maximum);
        };
        if (!inRange(vaVdst, 15) || !inRange(vaSdst, 7) || !inRange(vmVsrc, 7) ||
            !inRange(vaSsrc, 1) || !inRange(holdCnt, 1) || !inRange(vaVcc, 1) ||
            !inRange(saSdst, 1))
            fail(itemIndex, "S_WAIT_ALU modifier field is outside its hardware range");
        inst.addModifier(SWaitAluData(vaVdst, vaSdst, vaSsrc, holdCnt, vmVsrc, vaVcc, saSdst));
    }
}

bool hasPromotedEncoding(const HwInstDesc& desc) {
    return desc.promotedFormat != MicrocodeFormat::NONE && !desc.promotedFields.empty();
}

/// The promoted encoding of the table descriptor \p primary as a descriptor of its own: its
/// promoted fields and format, the `_e64` mnemonic and, for VOP3, the 64-bit encoding size.
/// One per primary descriptor, kept for the life of the process because IR built from it (an
/// AsmBuilder module) can outlive any call.
const HwInstDesc* promotedDescriptor(const HwInstDesc& primary) {
    struct Promoted {
        explicit Promoted(const HwInstDesc& primary)
            : mnemonic(std::string(primary.mnemonic) + "_e64"), desc(primary) {}
        std::string mnemonic;
        HwInstDesc desc;
    };
    // Python calls hold the GIL, which serializes access.
    static std::unordered_map<const HwInstDesc*, std::unique_ptr<Promoted>> cache;
    std::unique_ptr<Promoted>& slot = cache[&primary];
    if (!slot) {
        slot = std::make_unique<Promoted>(primary);
        slot->desc.mnemonic = slot->mnemonic.c_str();
        slot->desc.microcode = primary.promotedFormat;
        slot->desc.operandFields = primary.promotedFields;
        // Every promoted format the gfx tables define is the 64-bit VOP3.
        if (primary.promotedFormat == MicrocodeFormat::MC_VOP3) slot->desc.encoding = 64;
        slot->desc.promotedFormat = MicrocodeFormat::NONE;
        slot->desc.promotedFields = {};
    }
    return &slot->desc;
}

bool hasLabelField(const HwInstDesc& desc) {
    return std::any_of(desc.operandFields.begin(), desc.operandFields.end(),
                       [](const HwInstDesc::OperandFieldDesc& field) {
                           return field.fieldType == FieldType::label;
                       });
}

enum class SelectedForm { Primary, Promoted, NoReturn };

// MUBUF atomics are defined with vdata as a read-write destination, which the emitter
// prints as a returning atomic (th:TH_ATOMIC_RETURN). The ::noret form selects the
// fire-and-forget encoding instead: vdata only supplies the operand.
bool hasNoReturnForm(const std::string& mnemonic) {
    return mnemonic == "buffer_atomic_add_f32";
}

SelectedForm selectForm(const nb::dict& record, const HwInstDesc& desc,
                        const std::string& mnemonic, size_t itemIndex) {
    const std::string form = requireString(record, "form", itemIndex);
    const std::string primary = "gfx1250::" + mnemonic;
    if (hasNoReturnForm(mnemonic)) {
        if (form == primary + "::noret") return SelectedForm::NoReturn;
        fail(itemIndex, "instruction '" + mnemonic + "' is only supported in form '" + primary +
                            "::noret'");
    }
    if (form == primary) return SelectedForm::Primary;
    if (form == primary + "::vop3") {
        if (!hasPromotedEncoding(desc))
            fail(itemIndex, "form '" + form + "' requests a promoted encoding that instruction '" +
                                mnemonic + "' does not provide");
        return SelectedForm::Promoted;
    }

    const bool selectedExpansion =
        form == primary + "::terminator" ||
        (mnemonic == "s_nop" && form == primary + "::mode_hazard") ||
        (mnemonic == "s_set_vgpr_msb" && form == primary + "::generated") ||
        (mnemonic == "v_nop" && form == primary + "::from_repeat") ||
        ((mnemonic == "s_wait_dscnt" || mnemonic == "s_wait_kmcnt" ||
          mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_storecnt") &&
         form == primary + "::from_s_waitcnt");
    if (selectedExpansion) return SelectedForm::Primary;

    fail(itemIndex, "form '" + form + "' does not select instruction '" + mnemonic + "'");
}

void validateOperandCounts(const HwInstDesc& desc, size_t destCount, size_t srcCount,
                           bool promoted, size_t itemIndex) {
    size_t expectedDest = 0;
    size_t expectedSrc = 0;
    const auto fields = promoted ? desc.promotedFields : desc.operandFields;
    for (const auto& field : fields) {
        if (field.isDest)
            ++expectedDest;
        else
            ++expectedSrc;
    }
    const bool tensorSpec = (std::string_view(desc.mnemonic) == "tensor_load_to_lds" ||
                             std::string_view(desc.mnemonic) == "tensor_store_from_lds") &&
                            destCount == 0 && srcCount == 2;
    if (!tensorSpec && (destCount != expectedDest || srcCount != expectedSrc)) {
        fail(itemIndex, "instruction '" + std::string(desc.mnemonic) + "' expects " +
                            std::to_string(expectedDest) + " dst and " +
                            std::to_string(expectedSrc) + " src operands, got " +
                            std::to_string(destCount) + " dst and " +
                            std::to_string(srcCount) + " src");
    }
}

bool isScalarRegister(RegType type) {
    return type == RegType::S || type == RegType::SCC || type == RegType::VCC ||
           type == RegType::VCC_LO || type == RegType::VCC_HI || type == RegType::EXEC ||
           type == RegType::EXEC_LO || type == RegType::EXEC_HI;
}

bool isM0(const StinkyRegister& operand) {
    return operand.dataType == StinkyRegister::Type::Register && operand.reg.type == RegType::M &&
           operand.reg.idx == 0 && operand.reg.num == 1;
}

bool fieldRequiresVgpr(FieldType type) {
    return type == FieldType::vgpr || type == FieldType::src_vgpr;
}

bool fieldRequiresScalar(FieldType type) {
    return type == FieldType::sreg || type == FieldType::sreg_m0 ||
           type == FieldType::sgpr || type == FieldType::sdst || type == FieldType::ssrc;
}

bool fieldRequiresPhysicalScalar(FieldType type) {
    return type == FieldType::sreg || type == FieldType::sreg_m0 ||
           type == FieldType::sgpr || type == FieldType::sdst;
}

bool fieldRequiresInteger(FieldType type) {
    return type == FieldType::simm16 || type == FieldType::simm32 ||
           type == FieldType::simm24 || type == FieldType::simm5 ||
           type == FieldType::set_vgpr_msb || type == FieldType::sleep ||
           type == FieldType::ssrc_barrier_id || type == FieldType::delay ||
           type == FieldType::smem_offset_nok ||
           type == FieldType::wait_alu || type == FieldType::wait_mem_ds;
}

bool isNamedScalarSpecial(const std::string& value) {
    constexpr std::string_view prefix = "ttmp";
    if (!value.starts_with(prefix) || value.size() == prefix.size()) return false;
    if (!std::all_of(value.begin() + static_cast<std::ptrdiff_t>(prefix.size()), value.end(),
                     [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
        return false;
    unsigned index = 0;
    for (auto it = value.begin() + static_cast<std::ptrdiff_t>(prefix.size()); it != value.end(); ++it)
        index = index * 10 + static_cast<unsigned>(*it - '0');
    return index < 16;
}

bool allowsTtmp(const HwInstDesc& desc, bool isDest, size_t operandIndex,
                FieldType fieldType) {
    if (isDest || operandIndex != 0 || fieldType != FieldType::ssrc) return false;
    const std::string_view mnemonic(desc.mnemonic);
    return mnemonic == "s_and_b32" || mnemonic == "s_bfe_u32" ||
           mnemonic == "s_mov_b32";
}

bool allowsM0(const HwInstDesc& desc, bool isDest, size_t operandIndex,
              FieldType fieldType) {
    const std::string_view mnemonic(desc.mnemonic);
    if (mnemonic == "s_mov_b32")
        return isDest && operandIndex == 0 && fieldType == FieldType::sdst;
    if (mnemonic == "s_prefetch_inst")
        return !isDest && operandIndex == 2 && fieldType == FieldType::smem_offset_nok;
    return fieldType == FieldType::sreg_m0;
}

// v_mad_co_u64_u32 writes its carry to an SGPR/VCC or discards it with null.
bool isNullCarryDestination(const StinkyRegister& operand, const HwInstDesc& desc,
                            size_t operandIndex) {
    return operand.dataType == StinkyRegister::Type::LiteralString &&
           operand.literalValue == "null" && operandIndex == 1 &&
           std::string_view(desc.mnemonic) == "v_mad_co_u64_u32";
}

void validateRegisterRange(const StinkyRegister& operand, const ArchHelper::ArchInfo& archInfo,
                           const std::string& location, size_t itemIndex,
                           bool allowAuditedVgprBoundaryCrossing = false) {
    if (operand.reg.num == 0) fail(itemIndex, location + " register count must be positive");
    if (operand.reg.offset != 0)
        fail(itemIndex, location +
                            " must already be low-bit materialized; Register.offset must be zero");

    const uint64_t end = static_cast<uint64_t>(operand.reg.idx) + operand.reg.num;
    if (operand.reg.type == RegType::V && end > archInfo.maxVGPR &&
        !allowAuditedVgprBoundaryCrossing)
        fail(itemIndex, location + " VGPR range exceeds the directly encoded target range");
    if (operand.reg.type == RegType::S && end > archInfo.maxSGPR)
        fail(itemIndex, location + " SGPR range exceeds the target range");
    if ((operand.reg.type == RegType::A || operand.reg.type == RegType::ACC ||
         operand.reg.type == RegType::AGPR) &&
        (archInfo.maxAGPR == 0 || end > archInfo.maxAGPR))
        fail(itemIndex, location + " AGPR range is unsupported on this target");
}

void validateOperand(const StinkyRegister& operand, const HwInstDesc& desc,
                     const HwInstDesc::OperandFieldDesc& field,
                     const ArchHelper::ArchInfo& archInfo, GfxArchID arch, bool isDest,
                     size_t operandIndex, const std::string& location, size_t itemIndex) {
    if (operand.dataType == StinkyRegister::Type::Invalid)
        fail(itemIndex, location + " cannot be an invalid/null Register");
    if (operand.dataType == StinkyRegister::Type::HwReg) {
        if (field.fieldType != FieldType::hwreg)
            fail(itemIndex, location + " hwreg operand is only valid for a hardware-register field");
        if (operand.hwreg.size == 0 || operand.hwreg.offset + operand.hwreg.size > 32)
            fail(itemIndex, location + " hwreg slice must fit in 32 bits");
        if (HwReg::idToName(arch, operand.hwreg.id).empty())
            fail(itemIndex, location + " hwreg id is not defined for the selected target");
        return;
    }
    if (field.fieldType == FieldType::hwreg)
        fail(itemIndex, location + " hardware-register field requires a structured hwreg operand");
    if (field.isDest && isNullCarryDestination(operand, desc, operandIndex)) return;
    if (field.isDest && operand.dataType != StinkyRegister::Type::Register)
        fail(itemIndex, location + " destination must be a physical register");

    if (operand.dataType == StinkyRegister::Type::LiteralString) {
        if (field.fieldType == FieldType::label) {
            if (!isLabelName(operand.literalValue))
                fail(itemIndex, location + " is not a valid label name");
            return;
        }
        const std::string_view mnemonic(desc.mnemonic ? desc.mnemonic : "");
        if (operand.literalValue == "off" && field.encodeField == EncodeField::saddr &&
            (mnemonic == "flat_load_b32" || mnemonic == "flat_store_b32"))
            return;
        if (!isNamedScalarSpecial(operand.literalValue) ||
            !allowsTtmp(desc, isDest, operandIndex, field.fieldType))
            fail(itemIndex, location +
                                " string literals are only allowed for an audited TTMP scalar "
                                "source field or a compatible typed label");
        return;
    }

    if (field.fieldType == FieldType::label)
        fail(itemIndex, location + " label field requires a string target");

    if (field.fieldType == FieldType::sreg_m0) {
        // Buffer soffset: one SGPR, m0, or zero, which the emitter writes as null. The
        // field only names a register, so it has no room for negate/absolute flags.
        if (operand.dataType == StinkyRegister::Type::LiteralInt) {
            if (operand.literalInt != 0)
                fail(itemIndex, location + " soffset immediate must be 0 (null)");
            return;
        }
        if (!isM0(operand)) {
            if (operand.dataType != StinkyRegister::Type::Register ||
                operand.reg.type != RegType::S || operand.reg.num != 1)
                fail(itemIndex, location + " soffset requires one SGPR, m0 or 0 (null)");
            validateRegisterRange(operand, archInfo, location, itemIndex);
        }
        if (operand.reg.isMinus || operand.reg.isAbs)
            fail(itemIndex, location + " soffset does not support negate/absolute flags");
        return;
    }

    if (isM0(operand)) {
        if (!allowsM0(desc, isDest, operandIndex, field.fieldType))
            fail(itemIndex, location + " m0 is not valid for this operand field");
        return;
    }

    if (field.fieldType == FieldType::smem_offset) {
        if (operand.dataType == StinkyRegister::Type::LiteralInt) return;
        if (operand.dataType != StinkyRegister::Type::Register ||
            operand.reg.type != RegType::S)
            fail(itemIndex, location + " SMEM offset requires an integer literal or SGPR");
        validateRegisterRange(operand, archInfo, location, itemIndex);
        if (operand.reg.num != 1)
            fail(itemIndex, location + " SMEM offset SGPR must have register width 1");
        return;
    }

    if (fieldRequiresInteger(field.fieldType)) {
        if (operand.dataType != StinkyRegister::Type::LiteralInt)
            fail(itemIndex, location + " immediate field requires an integer Register literal");
        const int64_t value = operand.literalInt;
        if (field.fieldType == FieldType::set_vgpr_msb && (value < 0 || value > 0xffff))
            fail(itemIndex, location + " s_set_vgpr_msb immediate must be in [0, 65535]");
        if (field.fieldType == FieldType::simm16 && (value < -32768 || value > 65535))
            fail(itemIndex, location + " 16-bit immediate is out of range");
        if (field.fieldType == FieldType::simm24 &&
            (value < -(1 << 23) || value > (1 << 24) - 1))
            fail(itemIndex, location + " 24-bit signed/raw immediate is out of range");
        if (field.fieldType == FieldType::simm5 && (value < 0 || value > 31))
            fail(itemIndex, location + " 5-bit immediate is out of range");
        return;
    }

    if (operand.dataType != StinkyRegister::Type::Register) {
        if (fieldRequiresVgpr(field.fieldType) || fieldRequiresPhysicalScalar(field.fieldType) ||
            field.fieldType == FieldType::vcc || field.fieldType == FieldType::exec)
            fail(itemIndex, location + " field requires a physical register");
        return;
    }

    const bool auditedDsLoadB128BoundaryCrossing =
        std::string_view(desc.mnemonic) == "ds_load_b128" && isDest && operandIndex == 0 &&
        operand.reg.type == RegType::V && operand.reg.idx == 254 && operand.reg.num == 4;
    validateRegisterRange(operand, archInfo, location, itemIndex,
                          auditedDsLoadB128BoundaryCrossing);
    const bool auditedNegatedVfmaSrc0 =
        std::string_view(desc.mnemonic) == "v_fma_f32" && !isDest && operandIndex == 0 &&
        operand.reg.type == RegType::V && operand.reg.isMinus && !operand.reg.isAbs;
    if ((operand.reg.isMinus || operand.reg.isAbs) && !auditedNegatedVfmaSrc0)
        fail(itemIndex, location +
                            " per-operand negate/absolute flags are unsupported outside the "
                            "audited v_fma_f32 VGPR src0 form");
    if (fieldRequiresVgpr(field.fieldType) && operand.reg.type != RegType::V)
        fail(itemIndex, location + " field requires a VGPR");
    if (fieldRequiresScalar(field.fieldType) && field.fieldType != FieldType::sreg_m0 &&
        !isScalarRegister(operand.reg.type))
        fail(itemIndex, location + " field requires a scalar register");
    if (desc.microcode == MicrocodeFormat::MC_VOPC && isDest && operandIndex == 0 &&
        operand.reg.type != RegType::S && operand.reg.type != RegType::VCC &&
        operand.reg.type != RegType::VCC_LO && operand.reg.type != RegType::VCC_HI)
        fail(itemIndex, location + " VOPC destination requires an SGPR or VCC register");
    if (field.fieldType == FieldType::vcc && operand.reg.type != RegType::VCC &&
        operand.reg.type != RegType::VCC_LO && operand.reg.type != RegType::VCC_HI)
        fail(itemIndex, location + " field requires VCC");
    if (field.fieldType == FieldType::exec && operand.reg.type != RegType::EXEC &&
        operand.reg.type != RegType::EXEC_LO && operand.reg.type != RegType::EXEC_HI)
        fail(itemIndex, location + " field requires EXEC");

    if (field.fieldSizeBits != 0) {
        const uint16_t expectedCount = std::max<uint16_t>(1, (field.fieldSizeBits + 31) / 32);
        const bool wave32M64 = field.isM64 && archInfo.waveFrontSize == 32 &&
                               expectedCount == 2 && operand.reg.num == 1;
        if (operand.reg.num != expectedCount && !wave32M64)
            fail(itemIndex, location + " has register width " + std::to_string(operand.reg.num) +
                                ", expected " + std::to_string(expectedCount));
    }
}

void validateOperands(const HwInstDesc& desc, const std::vector<StinkyRegister>& dests,
                      const std::vector<StinkyRegister>& srcs,
                      const ArchHelper::ArchInfo& archInfo, GfxArchID arch,
                      bool promoted, size_t itemIndex) {
    size_t destIndex = 0;
    size_t srcIndex = 0;
    const auto fields = promoted ? desc.promotedFields : desc.operandFields;
    for (const auto& field : fields) {
        const bool isDest = field.isDest;
        const size_t operandIndex = isDest ? destIndex++ : srcIndex++;
        const auto& operands = isDest ? dests : srcs;
        if (operandIndex >= operands.size()) {
            if ((std::string_view(desc.mnemonic) == "tensor_load_to_lds" ||
                 std::string_view(desc.mnemonic) == "tensor_store_from_lds") &&
                !isDest &&
                srcs.size() == 2)
                continue;
            fail(itemIndex, "internal operand-count validation mismatch");
        }
        validateOperand(operands.at(operandIndex), desc, field, archInfo, arch,
                        isDest, operandIndex,
                        std::string(isDest ? "dst[" : "src[") +
                            std::to_string(operandIndex) + "]",
                        itemIndex);
    }
}

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
}

bool isHexDigit(char c) {
    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

std::string validateCustomEncoding(std::string text, size_t itemIndex) {
    if (text.empty()) fail(itemIndex, "custom encoding text must not be empty");
    if (text.find('\0') != std::string::npos)
        fail(itemIndex, "custom encoding text must not contain NUL bytes");

    std::istringstream stream(text);
    std::string line;
    size_t lineNumber = 0;
    size_t encodedWords = 0;
    while (std::getline(stream, line)) {
        ++lineNumber;
        std::string stripped = trim(line);
        if (stripped.empty()) continue;
        constexpr std::string_view prefix = ".long 0x";
        if (!stripped.starts_with(prefix) || stripped.size() != prefix.size() + 8 ||
            !std::all_of(stripped.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                         stripped.end(), isHexDigit)) {
            fail(itemIndex, "custom line " + std::to_string(lineNumber) +
                                " must be exactly '.long 0x' followed by 8 hexadecimal digits");
        }
        ++encodedWords;
    }
    if (encodedWords == 0) fail(itemIndex, "custom encoding must contain at least one .long word");
    if (text.back() != '\n') text.push_back('\n');
    return text;
}

std::string validateStandaloneComment(const std::string& text, size_t itemIndex) {
    if (text.find_first_of("\r\n") != std::string::npos)
        fail(itemIndex, "field 'text' must be a single line");
    if (text.find('\0') != std::string::npos)
        fail(itemIndex, "field 'text' must not contain NUL bytes");
    return "// " + text + "\n";
}

std::string emitAsm(const std::string& archName, uint32_t waveSize, const nb::list& items) {
    const auto& archHelper = ArchHelper::getInstance();
    const auto* archInfo = archHelper.getArchInfo(archName);
    if (!archInfo)
        throw nb::value_error(("emit_asm: unsupported architecture '" + archName + "'").c_str());
    if (waveSize != archInfo->waveFrontSize) {
        throw nb::value_error(("emit_asm: wave_size " + std::to_string(waveSize) +
                               " does not match " + archName + " wave size " +
                               std::to_string(archInfo->waveFrontSize))
                                  .c_str());
    }
    const GfxArchID arch = archHelper.getGfxArchID(archName);

    // The no-return descriptors must outlive every IR object that refers to them,
    // including Function destruction.
    std::vector<std::unique_ptr<HwInstDesc>> selectedDescriptorStorage;
    std::deque<std::vector<HwInstDesc::OperandFieldDesc>> selectedFieldStorage;
    Function function("emit_asm");
    BasicBlock* block = function.createBasicBlock("entry");
    AsmIRBuilder builder(*block, arch);
    size_t typedInstructionCount = 0;
    bool requiresM0WidthBypass = false;
    std::unordered_set<std::string> labels;
    std::vector<std::pair<std::string, size_t>> branchTargets;

    size_t itemIndex = 0;
    for (nb::handle item : items) {
        if (!PyDict_Check(item.ptr())) fail(itemIndex, "record must be a dict");
        nb::dict record = nb::borrow<nb::dict>(item);
        const std::string kind = requireString(record, "kind", itemIndex);

        if (kind == "instruction") {
            rejectUnknownKeys(record, {"kind", "form", "mnemonic", "dst", "src", "target",
                                       "modifiers", "comment"}, itemIndex);
            const std::string mnemonic = requireString(record, "mnemonic", itemIndex);
            const auto& mnemonicMap = archInfo->getMnemonicToIsaOpcodeMap();
            auto opcodeIt = mnemonicMap.find(mnemonic);
            if (opcodeIt == mnemonicMap.end()) {
                std::string message = "unknown instruction '" + mnemonic;
                message += "' for ";
                message += archName;
                fail(itemIndex, message);
            }
            if (!isSupportedTypedInstruction(mnemonic))
                fail(itemIndex, "instruction '" + mnemonic +
                                    "' is unsupported by the T01 typed bridge allowlist");
            const HwInstDesc* desc = getMCIDByIsaOp(opcodeIt->second, arch);
            if (!desc || !desc->mnemonic)
                fail(itemIndex, "instruction '" + mnemonic + "' has no hardware descriptor");
            const SelectedForm selectedForm = selectForm(record, *desc, mnemonic, itemIndex);
            const bool promoted = selectedForm == SelectedForm::Promoted;
            if (selectedForm == SelectedForm::NoReturn) {
                // Same descriptor, with the read-write vdata destination demoted to a plain
                // source, so the instruction has no destination and prints no
                // th:TH_ATOMIC_RETURN.
                auto& fields = selectedFieldStorage.emplace_back(desc->operandFields.begin(),
                                                                 desc->operandFields.end());
                for (auto& field : fields) {
                    if (!field.isDest) continue;
                    field.isDest = 0;
                    field.isReadWrite = 0;
                }
                auto selected = std::make_unique<HwInstDesc>(*desc);
                selected->operandFields = {fields.data(), fields.size()};
                selectedDescriptorStorage.push_back(std::move(selected));
                desc = selectedDescriptorStorage.back().get();
            }

            auto dests = parseOperands(record, "dst", itemIndex);
            auto srcs = parseOperands(record, "src", itemIndex);
            nb::handle targetValue = find(record, "target");
            if (hasLabelField(*desc)) {
                if (!targetValue.is_valid())
                    fail(itemIndex, "branch instruction '" + mnemonic + "' requires 'target'");
                if (!srcs.empty())
                    fail(itemIndex, "branch instruction '" + mnemonic +
                                        " uses 'target'; explicit src operands are not allowed");
                if (!PyUnicode_Check(targetValue.ptr()))
                    fail(itemIndex, "field 'target' must be a string");
                std::string target = nb::cast<std::string>(targetValue);
                if (!isLabelName(target)) fail(itemIndex, "field 'target' is not a valid label name");
                srcs.emplace_back(target);
                branchTargets.emplace_back(target, itemIndex);
            } else if (targetValue.is_valid()) {
                fail(itemIndex,
                     "field 'target' is only valid for an instruction with a label operand");
            }
            validateOperandCounts(*desc, dests.size(), srcs.size(), promoted, itemIndex);
            validateOperands(*desc, dests, srcs, *archInfo, arch, promoted, itemIndex);
            requiresM0WidthBypass = requiresM0WidthBypass ||
                                    std::any_of(dests.begin(), dests.end(), isM0) ||
                                    std::any_of(srcs.begin(), srcs.end(), isM0);

            const HwInstDesc* selectedDesc = promoted ? promotedDescriptor(*desc) : desc;

            StinkyInstruction* inst = builder.create(selectedDesc);
            inst->setDestRegs(dests);
            inst->setSrcRegs(srcs);
            addModifiers(*inst, find(record, "modifiers"), itemIndex);
            if (nb::handle comment = find(record, "comment"); comment.is_valid()) {
                if (!PyUnicode_Check(comment.ptr()))
                    fail(itemIndex, "field 'comment' must be a string");
                std::string text = nb::cast<std::string>(comment);
                if (text.find_first_of("\r\n") != std::string::npos)
                    fail(itemIndex, "field 'comment' must be a single line");
                inst->addModifier(CommentData{text});
            }
            ++typedInstructionCount;
        } else if (kind == "label") {
            rejectUnknownKeys(record, {"kind", "name", "alignment"}, itemIndex);
            std::string name = requireString(record, "name", itemIndex);
            if (!isLabelName(name)) fail(itemIndex, "field 'name' is not a valid label name");
            if (!labels.insert(name).second) fail(itemIndex, "duplicate label '" + name + "'");
            int64_t alignment = 1;
            if (nb::handle value = find(record, "alignment"); value.is_valid())
                alignment = requireInteger(value, "alignment", itemIndex);
            if (alignment < 1 || alignment > std::numeric_limits<uint16_t>::max())
                fail(itemIndex, "field 'alignment' must be in [1, 65535]");
            if ((alignment & (alignment - 1)) != 0)
                fail(itemIndex, "field 'alignment' must be a power of two");
            builder.createLabel(name, static_cast<uint16_t>(alignment));
        } else if (kind == "alignment") {
            rejectUnknownKeys(record, {"kind", "bytes"}, itemIndex);
            int64_t alignment =
                requireInteger(require(record, "bytes", itemIndex), "bytes", itemIndex);
            if (alignment < 1 || alignment > std::numeric_limits<int32_t>::max())
                fail(itemIndex, "field 'bytes' must be in [1, 2147483647]");
            if ((alignment & (alignment - 1)) != 0)
                fail(itemIndex, "field 'bytes' must be a power of two");
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::ALIGN;
            directive->name = ".align";
            directive->intValue = alignment;
        } else if (kind == "comment") {
            rejectUnknownKeys(record, {"kind", "text"}, itemIndex);
            std::string text = validateStandaloneComment(
                requireString(record, "text", itemIndex), itemIndex);
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = std::move(text);
        } else if (kind == "blank") {
            rejectUnknownKeys(record, {"kind"}, itemIndex);
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = "\n";
        } else if (kind == "custom") {
            rejectUnknownKeys(record, {"kind", "text"}, itemIndex);
            std::string text =
                validateCustomEncoding(requireString(record, "text", itemIndex), itemIndex);
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = std::move(text);
        } else {
            fail(itemIndex, "unknown kind '" + kind + "'");
        }
        ++itemIndex;
    }

    for (const auto& [target, sourceItem] : branchTargets) {
        if (!labels.contains(target)) fail(sourceItem, "undefined branch target '" + target + "'");
    }

    if (typedInstructionCount > 0) {
        AsmVerifierConfig config;
        config.abortOnError = false;
        // The upstream verifier models scalar fields as SGPR-like and does not
        // include RegType::M.  validateOperands above has already checked every
        // descriptor field's bank, range, and width; isM0 additionally requires
        // the exact one-register m0 spelling before this narrow verifier bypass.
        config.checkRegisterWidths = !requiresM0WidthBypass;
        config.checkRegisterRanges = false;
        config.checkReadWriteOperands = false;
        std::string error = validateStinkyIR(function, config);
        if (!error.empty())
            throw nb::value_error(("emit_asm: assembly IR validation failed: " + error).c_str());
    }

    AsmEmitterOptions options;
    options.emitCycleInfo = false;
    options.emitBlankLines = false;
    options.useSymbolicNames = false;
    StinkyAsmEmitter emitter(options);
    return emitter.emit(function);
}

// ---------------------------------------------------------------------------------------------
// AsmBuilder: build Asm IR from Python, one call per node, for the optimization pipeline.
// ---------------------------------------------------------------------------------------------

/// Names the AsmBuilder method in the errors raised while it runs (see fail()).
class AsmBuilderMethodScope {
   public:
    explicit AsmBuilderMethodScope(const char* method) : previous_(tAsmBuilderMethod) {
        tAsmBuilderMethod = method;
    }
    ~AsmBuilderMethodScope() {
        tAsmBuilderMethod = previous_;
    }
    AsmBuilderMethodScope(const AsmBuilderMethodScope&) = delete;
    AsmBuilderMethodScope& operator=(const AsmBuilderMethodScope&) = delete;

   private:
    const char* previous_;
};

/// The Registers of an AsmBuilder list argument. Register numbers are full numbers, and
/// StinkyTofu's VGPR-MSB pass derives Register.offset from them, so a nonzero offset is an error.
std::vector<StinkyRegister> builderRegisters(const nb::list& values, const char* argument) {
    std::vector<StinkyRegister> result;
    result.reserve(values.size());
    for (nb::handle value : values) {
        const std::string where = std::string(argument) + "[" + std::to_string(result.size()) + "]";
        if (!nb::isinstance<StinkyRegister>(value)) fail(0, where + " must be a Register");
        StinkyRegister reg = nb::cast<StinkyRegister>(value);
        if (reg.dataType == StinkyRegister::Type::Register && reg.reg.offset != 0)
            fail(0, where +
                        " has a nonzero Register.offset; pass the full register number "
                        "(StinkyTofu derives the VGPR-MSB offset)");
        result.push_back(std::move(reg));
    }
    return result;
}

/// Cast one options value. A bool is not taken for an integer option, nor an integer for a bool.
template <typename T>
bool castModuleOption(nb::handle value, T& out) {
    if constexpr (std::is_same_v<T, bool>) {
        if (!PyBool_Check(value.ptr())) return false;
    } else if constexpr (std::is_integral_v<T>) {
        if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) return false;
    }
    return nb::try_cast<T>(value, out);
}

/// ModuleOptions from an AsmBuilder options dict. toStinkyTofuModule only warns about an unknown
/// key or a wrongly typed value; here both are errors, because either would run the pipeline
/// without a setting the caller asked for.
StinkyAsmModule::ModuleOptions builderModuleOptions(const nb::object& optionsObject) {
    StinkyAsmModule::ModuleOptions moduleOptions{};
    // The unset values lower_logical_module starts from.
    moduleOptions.SwInstructionPrefetchAbsBaseSgpr = -1;
    moduleOptions.DsReadOrder = -1;
    if (optionsObject.is_none()) return moduleOptions;
    if (!PyDict_Check(optionsObject.ptr())) fail(0, "'options' must be a dict or None");
    nb::dict options = nb::borrow<nb::dict>(optionsObject);
    std::unordered_set<std::string> known;
#define SET_BUILDER_MODULE_OPTION(name, type)                                   \
    known.insert(#name);                                                        \
    if (nb::handle value = find(options, #name);                                \
        value.is_valid() && !castModuleOption<type>(value, moduleOptions.name)) \
        fail(0, "option '" #name "' has the wrong type");
#define SET_BUILDER_MODULE_OPTION_WITH_DEFAULT(name, type, defaultValue) \
    SET_BUILDER_MODULE_OPTION(name, type)
    MODULE_OPTIONS_LIST(SET_BUILDER_MODULE_OPTION)
    MODULE_OPTIONS_WITH_DEFAULTS_LIST(SET_BUILDER_MODULE_OPTION_WITH_DEFAULT)
#undef SET_BUILDER_MODULE_OPTION_WITH_DEFAULT
#undef SET_BUILDER_MODULE_OPTION
    Py_ssize_t position = 0;
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    while (PyDict_Next(options.ptr(), &position, &key, &value)) {
        if (!PyUnicode_Check(key)) fail(0, "option keys must be strings");
        const std::string name = nb::cast<std::string>(nb::handle(key));
        if (!known.contains(name)) fail(0, "unknown option '" + name + "'");
    }
    return moduleOptions;
}

/// Mnemonics the gfx1250 descriptor table has but the assembler rejects, with the forms to use.
const char* gfx1250AssemblerReplacement(std::string_view mnemonic) {
    static constexpr std::pair<std::string_view, const char*> kRejected[] = {
        {"ds_load_b192", "ds_load_b128 and ds_load_b64"},
        {"ds_store_b192", "ds_store_b128 and ds_store_b64"},
        {"ds_store_b256", "two ds_store_b128"},
        {"s_waitcnt", "s_wait_loadcnt, s_wait_storecnt, s_wait_dscnt or s_wait_kmcnt"},
        {"s_barrier", "s_barrier_signal -1 and s_barrier_wait -1"},
    };
    for (const auto& [rejected, replacement] : kRejected)
        if (mnemonic == rejected) return replacement;
    return nullptr;
}

/// The number of source fields of \p desc, or of its promoted encoding.
size_t sourceFieldCount(const HwInstDesc& desc, bool promoted) {
    const auto& fields = promoted ? desc.promotedFields : desc.operandFields;
    return static_cast<size_t>(
        std::count_if(fields.begin(), fields.end(),
                      [](const HwInstDesc::OperandFieldDesc& field) { return !field.isDest; }));
}

/// The label instruction named \p name in \p block, or null.
StinkyInstruction* findLabel(BasicBlock& block, const std::string& name) {
    for (IRBase& node : block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == nullptr || !isLabel(*inst)) continue;
        if (const auto* data = inst->getModifier<LabelData>(); data && data->label == name)
            return inst;
    }
    return nullptr;
}

class AsmBuilder {
   public:
    AsmBuilder(const std::string& archName, const std::string& moduleName,
               const nb::object& options) {
        AsmBuilderMethodScope scope("__init__");
        const auto& archHelper = ArchHelper::getInstance();
        archInfo_ = archHelper.getArchInfo(archName);
        if (!archInfo_) fail(0, "unsupported architecture '" + archName + "'");
        arch_ = archHelper.getGfxArchID(archName);
        archName_ = archName;
        const std::array<int, 3> triple{static_cast<int>(archInfo_->major),
                                        static_cast<int>(archInfo_->minor),
                                        static_cast<int>(archInfo_->stepping)};
        const auto* pipeline = BackendRegistry::getArchPipeline(triple);
        if (pipeline == nullptr || !pipeline->builder)
            fail(0, "no optimization pipeline is registered for " + archName);

        StinkyAsmModule::ModuleOptions moduleOptions = builderModuleOptions(options);
        if (!moduleOptions.ArchName.empty() &&
            archHelper.getArchInfo(moduleOptions.ArchName) != archInfo_)
            fail(0, "option 'ArchName' names '" + moduleOptions.ArchName + "', not " + archName);
        if (moduleOptions.wavefrontSize != 0 &&
            moduleOptions.wavefrontSize != static_cast<int>(archInfo_->waveFrontSize))
            fail(0, "option 'wavefrontSize' is " + std::to_string(moduleOptions.wavefrontSize) +
                        ", but " + archName + " runs wave" +
                        std::to_string(archInfo_->waveFrontSize));
        if (const auto keep = moduleOptions.keepAuthored(); keep.hasError())
            fail(0, keep.getError());

        module_ = std::make_shared<StinkyAsmModule>(moduleName, triple, moduleOptions);
        for (const auto& groupName : pipeline->groupNames) module_->addGroup(groupName);
        builder_ = std::make_unique<AsmIRBuilder>(*module_->getFunction().getEntryBlock(), arch_);
    }

    std::vector<int> readWriteDests(const std::string& mnemonic, bool promoted) const {
        AsmBuilderMethodScope scope("read_write_dests");
        const HwInstDesc* desc = lookup(mnemonic);
        if (promoted && !hasPromotedEncoding(*desc))
            fail(0, "instruction '" + mnemonic + "' has no promoted encoding");
        std::vector<int> result;
        int position = 0;
        for (const auto& field : promoted ? desc->promotedFields : desc->operandFields) {
            if (!field.isDest) continue;
            if (field.isReadWrite) result.push_back(position);
            ++position;
        }
        return result;
    }

    void instruction(const std::string& mnemonic, const nb::list& dst, const nb::list& src,
                     bool promoted, const nb::list& implicitSrc, const nb::object& modifiers,
                     const nb::object& target, const std::string& comment, bool legalize) {
        AsmBuilderMethodScope scope("instruction");
        checkOpen();
        // A rejected call leaves nothing behind: the checks run before the instruction is
        // created, except addModifiers', which need it and erase it when they fail, and the
        // bookkeeping comes last.
        const HwInstDesc* desc = lookup(mnemonic);
        if (archInfo_->major == 12 && archInfo_->minor == 5) {
            if (const char* replacement = gfx1250AssemblerReplacement(mnemonic))
                fail(0, "the assembler rejects '" + mnemonic + "' on " + archName_ + "; use " +
                            replacement);
        }
        if (promoted && !hasPromotedEncoding(*desc))
            fail(0, "instruction '" + mnemonic + "' has no promoted encoding");

        std::vector<StinkyRegister> dests = builderRegisters(dst, "dst");
        std::vector<StinkyRegister> srcs = builderRegisters(src, "src");
        std::optional<std::string> branchTarget;
        if (hasLabelField(*desc)) {
            if (target.is_none())
                fail(0, "branch instruction '" + mnemonic + "' requires 'target'");
            if (!srcs.empty())
                fail(0, "branch instruction '" + mnemonic + "' takes 'target', not src operands");
            if (!PyUnicode_Check(target.ptr())) fail(0, "'target' must be a string");
            branchTarget = nb::cast<std::string>(target);
            if (!isLabelName(*branchTarget)) fail(0, "'target' is not a valid label name");
            srcs.emplace_back(*branchTarget);
        } else if (!target.is_none()) {
            fail(0, "'target' is only valid for an instruction with a label operand");
        }
        validateOperandCounts(*desc, dests.size(), srcs.size(), promoted, 0);
        validateOperands(*desc, dests, srcs, *archInfo_, arch_, promoted, 0);
        const std::vector<StinkyRegister> implicitSrcs = implicitSources(implicitSrc);
        // On a form with fewer sources than source fields (the two-source tensor_load_to_lds
        // form), the verifier and every pass would read an appended register as the next
        // field's operand.
        if (const size_t fields = sourceFieldCount(*desc, promoted);
            !implicitSrcs.empty() && srcs.size() < fields)
            fail(0, "implicit_src is not accepted for '" + mnemonic + "' with " +
                        std::to_string(srcs.size()) + " of its " + std::to_string(fields) +
                        " source operands");
        if (comment.find_first_of(std::string_view("\r\n\0", 3)) != std::string::npos)
            fail(0, "'comment' must be a single line without NUL bytes");
        if (!modifiers.is_none() && !PyDict_Check(modifiers.ptr()))
            fail(0, "'modifiers' must be a dict or None");

        StinkyInstruction* inst = builder_->create(promoted ? promotedDescriptor(*desc) : desc);
        try {
            inst->setDestRegs(dests);
            inst->setSrcRegs(srcs);
            addModifiers(*inst, modifiers.is_none() ? nb::handle() : nb::handle(modifiers), 0);
            if (!comment.empty()) inst->addModifier(CommentData{comment});
            const size_t explicitSrcCount = inst->getSrcRegs().size();
            if (legalize) {
                // The operand steps of the rocisa conversion's legalizeInstruction, in its
                // order: SCC/VCC/EXEC accesses the descriptor declares, then the hidden source
                // of each read-write destination (after the modifiers, since op_sel makes a
                // narrow destination read-write). No instruction is split or rewritten.
                legalizeImplicitSpecialRegisters(inst, archInfo_->waveFrontSize);
                legalizeReadWriteSources(inst);
            }
            bool appendedImplicit = false;
            for (const StinkyRegister& reg : implicitSrcs) {
                const auto& present = inst->getSrcRegs();
                if (std::find(present.begin(), present.end(), reg) != present.end()) continue;
                inst->addSrcReg(reg);
                appendedImplicit = true;
            }
            // Without the count, the emitter would print the appended registers as operands of
            // an instruction whose descriptor has no operand fields (s_wait_idle, s_endpgm).
            if (appendedImplicit) inst->addModifier(ExplicitSrcCountData(explicitSrcCount));
        } catch (...) {
            inst->erase();
            throw;
        }
        if (branchTarget) branchTargets_.emplace_back(*branchTarget, count_);
        allLegalized_ = allLegalized_ && legalize;
        requiresM0WidthBypass_ = requiresM0WidthBypass_ ||
                                 std::any_of(dests.begin(), dests.end(), isM0) ||
                                 std::any_of(srcs.begin(), srcs.end(), isM0);
        ++instructions_;
        ++count_;
    }

    void label(const std::string& name, int64_t alignment) {
        AsmBuilderMethodScope scope("label");
        checkOpen();
        if (!isLabelName(name)) fail(0, "'" + name + "' is not a valid label name");
        if (alignment < 1 || alignment > std::numeric_limits<uint16_t>::max() ||
            (alignment & (alignment - 1)) != 0)
            fail(0, "label alignment must be a power of two in [1, 65535]");
        if (labels_.contains(name)) fail(0, "duplicate label '" + name + "'");
        builder_->createLabel(name, static_cast<uint16_t>(alignment));
        labels_.insert(name);
        ++count_;
    }

    void comment(const std::string& text) {
        AsmBuilderMethodScope scope("comment");
        checkOpen();
        std::string value = validateStandaloneComment(text, 0);
        AsmDirective* directive = builder_->createIR<AsmDirective>();
        directive->kind = AsmDirectiveKind::TEXTBLOCK;
        directive->value = std::move(value);
        ++count_;
    }

    void blank() {
        AsmBuilderMethodScope scope("blank");
        checkOpen();
        AsmDirective* directive = builder_->createIR<AsmDirective>();
        directive->kind = AsmDirectiveKind::TEXTBLOCK;
        directive->value = "\n";
        ++count_;
    }

    nb::dict setGroupRange(const std::string& group, const std::string& firstLabel,
                           const std::string& endLabel, bool endInclusive) {
        AsmBuilderMethodScope scope("set_group_range");
        if (!module_->hasGroup(group)) fail(0, "the pipeline registered no group '" + group + "'");
        // Looked up now rather than kept from label(): the pipeline moves labels out of the
        // entry block and can delete them, so a kept pointer could dangle.
        BasicBlock& entry = *module_->getFunction().getEntryBlock();
        auto entryLabel = [&](const std::string& name) {
            StinkyInstruction* node = findLabel(entry, name);
            if (node == nullptr)
                fail(0, "label '" + name + "' is not in the entry block (" +
                            (labels_.contains(name) ? "runOptimizationPipeline() already ran"
                                                    : "no such label was built") +
                            ")");
            return node;
        };
        StinkyInstruction* first = entryLabel(firstLabel);
        StinkyInstruction* endNode = entryLabel(endLabel);
        // Walk forward from first to end_label: end_label must come after first_label.
        IRBase* last = nullptr;
        for (IRBase* node = first; node != endNode; node = node->getNext()) {
            if (node == nullptr) fail(0, "end_label does not follow first_label");
            if (dyn_cast<StinkyInstruction>(node)) last = node;
        }
        if (endInclusive) last = endNode;
        // last is null when first_label is end_label and end_inclusive is false.
        size_t nodes = 0, instructions = 0, labels = 0, directives = 0;
        if (last != nullptr) {
            for (IRBase* node = first;; node = node->getNext()) {
                ++nodes;
                if (auto* inst = dyn_cast<StinkyInstruction>(node)) {
                    if (isLabel(*inst))
                        ++labels;
                    else
                        ++instructions;
                } else {
                    ++directives;
                }
                if (node == last) break;
            }
        }
        // Checked before the range is stored, so a rejected call changes nothing.
        if (instructions == 0) fail(0, "the range holds no instruction besides labels");
        module_->setGroupRange(group, IntrusiveListIterator<IRBase>(first),
                               IntrusiveListIterator<IRBase>(last));
        auto* lastInst = cast<StinkyInstruction>(last);
        nb::dict info;
        info["group"] = group;
        info["first"] = firstLabel;
        info["last_mnemonic"] = std::string(lastInst->getHwInstDesc()->mnemonic);
        if (const auto* data = lastInst->getModifier<LabelData>()) info["last_label"] = data->label;
        info["nodes"] = nodes;
        info["instructions"] = instructions;
        info["labels"] = labels;
        info["directives"] = directives;
        return info;
    }

    std::shared_ptr<StinkyAsmModule> finish() {
        AsmBuilderMethodScope scope("finish");
        checkOpen();
        for (const auto& [target, node] : branchTargets_)
            if (!labels_.contains(target))
                fail(0, "undefined branch target '" + target + "' (branch at node " +
                            std::to_string(node) + ")");
        if (instructions_ > 0) {
            // As emit_asm verifies its records. Read-write sources are checked only when every
            // instruction was legalized, since legalize adds them.
            AsmVerifierConfig config;
            config.abortOnError = false;
            config.checkRegisterWidths = !requiresM0WidthBypass_;
            config.checkRegisterRanges = false;
            config.checkReadWriteOperands = allLegalized_;
            const std::string error = validateStinkyIR(module_->getFunction(), config);
            if (!error.empty()) fail(0, "assembly IR validation failed: " + error);
        }
        finished_ = true;
        return module_;
    }

   private:
    const HwInstDesc* lookup(const std::string& mnemonic) const {
        const auto& mnemonicMap = archInfo_->getMnemonicToIsaOpcodeMap();
        auto opcodeIt = mnemonicMap.find(mnemonic);
        if (opcodeIt == mnemonicMap.end())
            fail(0, "unknown instruction '" + mnemonic + "' for " + archName_);
        const HwInstDesc* desc = getMCIDByIsaOp(opcodeIt->second, arch_);
        if (!desc || !desc->mnemonic)
            fail(0, "instruction '" + mnemonic + "' has no hardware descriptor");
        return desc;
    }

    /// implicit_src: physical SGPR or VGPR ranges the target has, without flags.
    std::vector<StinkyRegister> implicitSources(const nb::list& values) const {
        std::vector<StinkyRegister> regs = builderRegisters(values, "implicit_src");
        for (size_t i = 0; i < regs.size(); ++i) {
            const StinkyRegister& reg = regs[i];
            const std::string where = "implicit_src[" + std::to_string(i) + "]";
            if (reg.dataType != StinkyRegister::Type::Register ||
                (reg.reg.type != RegType::S && reg.reg.type != RegType::V))
                fail(0, where + " must be an SGPR or VGPR register");
            if (reg.reg.isMinus || reg.reg.isAbs)
                fail(0, where + " cannot carry negate/absolute flags");
            validateRegisterRange(reg, *archInfo_, where, 0);
        }
        return regs;
    }

    void checkOpen() const {
        if (finished_) fail(0, "finish() was already called");
    }

    const ArchHelper::ArchInfo* archInfo_ = nullptr;
    GfxArchID arch_{};
    std::string archName_;
    std::shared_ptr<StinkyAsmModule> module_;
    std::unique_ptr<AsmIRBuilder> builder_;
    std::unordered_set<std::string> labels_;
    std::vector<std::pair<std::string, size_t>> branchTargets_;  // target, node index
    size_t count_ = 0;                                           // nodes built
    size_t instructions_ = 0;
    bool allLegalized_ = true;
    bool requiresM0WidthBypass_ = false;
    bool finished_ = false;
};

}  // namespace

// Called from python_bindings.cpp, which declares it, so it cannot have internal linkage.
// NOLINTNEXTLINE(misc-use-internal-linkage)
void init_asm_bridge(nb::module_& m) {
    nb::class_<AsmBuilder>(
        m, "AsmBuilder",
        R"doc(Build a StinkyTofu Asm IR module node by node for runOptimizationPipeline().

emit_asm prints already-selected records as text and runs no optimization pass. AsmBuilder
instead builds a StinkyAsmModule that StinkyTofu's pipeline can schedule and rewrite. Each call
checks its arguments against the target's instruction descriptor and either builds one node or
raises ValueError and builds nothing.

AsmBuilder does not select forms or encodings (the caller names the descriptor mnemonic and asks
for the promoted encoding explicitly), allocate registers, insert VGPR-MSB state (the pipeline
does), or split or rewrite instructions. Unlike emit_asm it has no allowlist: every mnemonic in
the target's table is accepted, except on gfx1250 the forms the assembler rejects (ds_load_b192,
ds_store_b192, ds_store_b256, s_waitcnt, s_barrier). VGPR numbers are full numbers (v300, not
v44 under an s_set_vgpr_msb).
)doc")
        .def(nb::init<const std::string&, const std::string&, const nb::object&>(), nb::arg("arch"),
             nb::arg("name"), nb::arg("options") = nb::none(),
             R"doc(Start an empty module.

Args:
    arch: target name, e.g. "gfx1250". The module takes the target's architecture triple and
        instruction groups (the group names its optimization pipeline registers).
    name: module (kernel) name.
    options: ModuleOptions as a dict keyed by field name (OptLevel, RegisterAllocation,
        KeepAuthored, ...), or None.

Raises:
    ValueError: an unknown architecture or one with no registered optimization pipeline; an
        options key that is not a ModuleOptions field, or a value of another type (bool and int
        are not interchangeable); ArchName or wavefrontSize disagreeing with arch (ArchName
        must be a target name that denotes arch's target; a feature suffix such as ":xnack-"
        is not part of one); a KeepAuthored token other than nop, mov and reuse.
)doc")
        .def("read_write_dests", &AsmBuilder::readWriteDests, nb::arg("mnemonic"),
             nb::arg("promoted") = false,
             R"doc(Destination positions the descriptor of mnemonic marks read-write.

Positions count the destinations only, from 0. With promoted=True the promoted encoding's
descriptor is read. Taken from the descriptor alone: a narrow destination that an
instruction's op_sel makes read-write is not included. Nothing is built.

Raises:
    ValueError: an unknown mnemonic, or promoted=True for one without a promoted encoding.
)doc")
        .def("instruction", &AsmBuilder::instruction, nb::arg("mnemonic"), nb::arg("dst"),
             nb::arg("src"), nb::kw_only(), nb::arg("promoted") = false,
             nb::arg("implicit_src") = nb::list(), nb::arg("modifiers") = nb::none(),
             nb::arg("target") = nb::none(), nb::arg("comment") = "", nb::arg("legalize") = true,
             R"doc(Append one instruction.

Args:
    mnemonic: descriptor mnemonic from the target's table.
    dst, src: Register operands in descriptor order, as emit_asm takes them (including the
        two-source tensor_load_to_lds / tensor_store_from_lds form). A VGPR is named by its
        full number, and Register.offset must be 0.
    promoted: build the promoted encoding (VOP3, printed with _e64) instead of the primary one.
    implicit_src: SGPR or VGPR registers the instruction is taken to read, for dependency
        tracking only, for example scalar-load destinations that must stay reserved until a
        wait. Appended after the other sources unless already among them; never printed.
        Not accepted with fewer src operands than the descriptor has source fields (the
        two-source tensor_load_to_lds form), where the passes would read them as operands.
    modifiers: the dict emit_asm accepts as a record's 'modifiers', or None.
    target: label name; required by, and only valid for, an instruction with a label operand.
        The label may be appended later.
    comment: single-line text printed after the instruction.
    legalize: add the sources StinkyTofu's passes expect: the SCC/VCC/EXEC accesses the
        descriptor declares, and the hidden source of each read-write destination. With
        legalize=False the instruction is built as given, and finish() then skips the
        read-write source check for the whole module.

Raises:
    ValueError: an unknown mnemonic or a form the assembler rejects; promoted=True without a
        promoted encoding; operands that do not fit the descriptor; a bad implicit_src entry,
        modifier, target or comment; any call after finish(). Nothing is built.
)doc")
        .def("label", &AsmBuilder::label, nb::arg("name"), nb::arg("alignment") = 1,
             R"doc(Append the label name, after `.align alignment` when alignment is above 1.

Raises:
    ValueError: an invalid or duplicate name, an alignment that is not a power of two in
        [1, 65535], or a call after finish(). Nothing is built.
)doc")
        .def("comment", &AsmBuilder::comment, nb::arg("text"),
             R"doc(Append a standalone comment line `// text`.

Raises:
    ValueError: text spans lines or holds a NUL byte, or a call after finish().
)doc")
        .def("blank", &AsmBuilder::blank, R"doc(Append an empty line.

Raises:
    ValueError: a call after finish().
)doc")
        .def("set_group_range", &AsmBuilder::setGroupRange, nb::arg("group"),
             nb::arg("first_label"), nb::arg("end_label"), nb::arg("end_inclusive") = false,
             R"doc(Give an instruction group a range of the module, by label.

The range starts at the label first_label and ends at the last instruction or label before the
label end_label; comment and blank nodes directly before end_label stay outside (a label's
alignment belongs to the label). With end_inclusive=True it ends at end_label itself. Only the
range is stored (StinkyAsmModule::setGroupRange); nothing is built or moved.

Allowed before or after finish(), but call it before runOptimizationPipeline(): the pipeline
moves labels out of the entry block, where they are looked up when this is called.

Args:
    group: an instruction group the target's pipeline registers, e.g. "loopWithPrefetch".

Returns:
    dict with group, first, last_mnemonic, last_label (when the range ends at a label) and the
    counts nodes, instructions, labels and directives the range holds.

Raises:
    ValueError: an unregistered group; a label not in the entry block; end_label not after
        first_label; a range that holds no instruction besides labels. Nothing is stored.
)doc")
        .def("finish", &AsmBuilder::finish,
             R"doc(Check the module, close the builder and return the StinkyAsmModule.

Every branch target must be a label of the module, and the instructions must pass the Asm IR
verifier (validateStinkyIR: register types and widths against the descriptor, and read-write
sources when every instruction was legalized). Afterwards every call that builds raises; the
module stays valid after the builder is gone.

Raises:
    ValueError: an undefined branch target or a verifier error (the builder then stays open), or
        a second call.
)doc");
    m.def("emit_asm", &emitAsm, nb::arg("arch"), nb::arg("wave_size"), nb::arg("items"),
          R"doc(Emit ordered, already-selected assembly records through temporary Asm IR.

Instruction records use existing Register values and exact descriptor mnemonics. The bridge
does not select forms, allocate registers, insert VGPR-MSB state, or run optimization passes.
Standalone comment and blank records become safe TEXTBLOCK directives; comments are always prefixed
with `// ` after rejecting multiline/NUL payloads, while blanks accept no payload. Custom records are
restricted to one or more 32-bit `.long 0xXXXXXXXX` encoding directives. Their bits are trusted output
from an upstream audited typed encoder; this bridge validates spelling and ordering, not the encoded
instruction semantics or control flow.
)doc");
}
