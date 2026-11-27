"""Tests for AsmBuilder, which builds Asm IR node by node for the optimization pipeline."""

import gc

import pytest
from stinkytofu import AsmBuilder, Register, emit_asm, sgpr, vgpr


def _lines(text):
    """Non-empty lines with each whitespace run made one space. emit_asm indents instructions and
    emitAssembly does not, which also moves the column trailing comments are aligned to.
    """
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _built(build, options=None, target="gfx1250"):
    builder = AsmBuilder(target, "kernel", options)
    build(builder)
    return builder.finish().emitAssembly()


# The gfx1250 target names; gfx1250-strict is the toolchain name of gfx1250v0.
TARGETS = ["gfx1250", "gfx1250v0", "gfx1250-strict"]


def _skip_unbuilt(target):
    try:
        AsmBuilder(target, "kernel")
    except ValueError as error:
        if "unsupported architecture" not in str(error):
            raise
        pytest.skip(f"{target} is not built (see STINKYTOFU_ARCHS_TO_BUILD)")


# Instruction records in emit_asm's shape; the parity test builds each through both paths.
PARITY_RECORDS = [
    {"kind": "label", "name": "entry", "alignment": 16},
    {
        "kind": "instruction",
        "mnemonic": "v_add_nc_u32",
        "dst": [vgpr(44)],
        "src": [vgpr(45), vgpr(46)],
    },
    {
        "kind": "instruction",
        "mnemonic": "v_add_nc_u64",
        "dst": [vgpr(48, 2)],
        "src": [vgpr(50, 2), vgpr(52, 2)],
    },
    {
        "kind": "instruction",
        "mnemonic": "v_wmma_f32_16x16x32_bf16",
        "dst": [vgpr(64, 8)],
        "src": [vgpr(72, 8), vgpr(80, 8), vgpr(64, 8)],
        "modifiers": {"mfma": {"reuse_a": True}},
    },
    {
        "kind": "instruction",
        "mnemonic": "ds_load_b128",
        "dst": [vgpr(88, 4)],
        "src": [vgpr(4)],
        "modifiers": {"ds": {"offset": 32}},
        "comment": "tile load",
    },
    {
        "kind": "instruction",
        "form": "gfx1250::v_add_nc_u32::vop3",
        "mnemonic": "v_add_nc_u32",
        "dst": [vgpr(0)],
        "src": [vgpr(1), sgpr(2)],
    },
    {
        "kind": "instruction",
        "form": "gfx1250::v_cmp_eq_f32::vop3",
        "mnemonic": "v_cmp_eq_f32",
        "dst": [sgpr(12)],
        "src": [vgpr(13), Register(0)],
    },
    {
        "kind": "instruction",
        "mnemonic": "s_load_b32",
        "dst": [sgpr(0)],
        "src": [sgpr(2, 2), Register(0)],
        "modifiers": {"smem": {"offset": 4}},
    },
    {
        "kind": "instruction",
        "mnemonic": "s_addk_co_i32",
        "dst": [sgpr(5)],
        "src": [Register(1)],
    },
    {"kind": "instruction", "mnemonic": "s_wait_kmcnt", "src": [Register(0)]},
    {"kind": "instruction", "mnemonic": "s_cbranch_scc0", "target": "done"},
    {"kind": "comment", "text": "standalone comment"},
    {"kind": "blank"},
    {"kind": "label", "name": "done"},
    {"kind": "instruction", "mnemonic": "s_endpgm"},
]


def _build_record(builder, record):
    kind = record["kind"]
    if kind == "label":
        builder.label(record["name"], record.get("alignment", 1))
    elif kind == "comment":
        builder.comment(record["text"])
    elif kind == "blank":
        builder.blank()
    else:
        builder.instruction(
            record["mnemonic"],
            record.get("dst", []),
            record.get("src", []),
            promoted=record.get("form", "").endswith("::vop3"),
            modifiers=record.get("modifiers"),
            target=record.get("target"),
            comment=record.get("comment", ""),
        )


def _emit_record(record):
    record = dict(record)
    if record["kind"] == "instruction":
        record.setdefault("form", f"gfx1250::{record['mnemonic']}")
    return record


def _assert_parity(target, records):
    reference = emit_asm(target, 32, [_emit_record(record) for record in records])

    def build(builder):
        for record in records:
            _build_record(builder, record)

    assert _lines(_built(build, target=target)) == _lines(reference)


@pytest.mark.parametrize("target", TARGETS)
def test_builder_prints_what_emit_asm_prints(target):
    _skip_unbuilt(target)
    records = PARITY_RECORDS
    if target != "gfx1250":
        # The gfx1250v0 table has no s_addk_co_i32.
        records = [r for r in records if r.get("mnemonic") != "s_addk_co_i32"]
    _assert_parity(target, records)


@pytest.mark.parametrize("target", TARGETS)
def test_builder_takes_m0_operands_like_emit_asm(target):
    # An m0 operand turns off finish()'s register-width check, as it does emit_asm's; kept out
    # of PARITY_RECORDS so that module stays width-checked.
    _skip_unbuilt(target)
    _assert_parity(
        target,
        [
            {
                "kind": "instruction",
                "mnemonic": "s_mov_b32",
                "dst": [Register("m", 0, 1)],
                "src": [sgpr(4)],
            },
            {"kind": "instruction", "mnemonic": "s_endpgm"},
        ],
    )


def test_legalize_adds_no_printed_operand():
    def build(legalize):
        def body(builder):
            for record in PARITY_RECORDS:
                if record["kind"] != "instruction":
                    _build_record(builder, record)
                    continue
                builder.instruction(
                    record["mnemonic"],
                    record.get("dst", []),
                    record.get("src", []),
                    promoted=record.get("form", "").endswith("::vop3"),
                    modifiers=record.get("modifiers"),
                    target=record.get("target"),
                    comment=record.get("comment", ""),
                    legalize=legalize,
                )

        return _built(body)

    assert build(True) == build(False)


@pytest.mark.parametrize(
    ("mnemonic", "dst", "src", "implicit", "printed"),
    [
        # No operand fields: the emitter would otherwise print every source.
        ("s_wait_idle", [], [], [sgpr(4, 2)], "s_wait_idle"),
        ("s_endpgm", [], [], [sgpr(6)], "s_endpgm"),
        # Controls: these descriptors have source fields, which alone limit what is printed.
        ("s_wait_kmcnt", [], [Register(0)], [sgpr(20, 2), vgpr(300)], "s_wait_kmcnt 0"),
        (
            "tensor_store_from_lds",
            [],
            [sgpr(8, 4), sgpr(12, 8)],
            [sgpr(40, 4), vgpr(300)],
            "tensor_store_from_lds s[8:11], s[12:19]",
        ),
    ],
)
@pytest.mark.parametrize("run_pipeline", [False, True], ids=["built", "after_pipeline"])
def test_implicit_sources_are_never_printed(
    mnemonic, dst, src, implicit, printed, run_pipeline
):
    builder = AsmBuilder("gfx1250", "kernel", {"OptLevel": 0})
    builder.instruction(mnemonic, dst, src, implicit_src=implicit)
    if mnemonic != "s_endpgm":
        builder.instruction("s_endpgm", [], [])
    module = builder.finish()
    if run_pipeline:
        module.runOptimizationPipeline()
    assert printed in _lines(module.emitAssembly())


@pytest.mark.parametrize(
    ("implicit", "message"),
    [
        ([sgpr(200, 2)], r"implicit_src\[0\] SGPR range exceeds"),
        ([vgpr(5000)], r"implicit_src\[0\] VGPR range exceeds"),
        ([Register()], r"implicit_src\[0\] must be an SGPR or VGPR"),
        ([Register(5)], r"implicit_src\[0\] must be an SGPR or VGPR"),
        ([Register("vcc_lo", 0, 1)], r"implicit_src\[0\] must be an SGPR or VGPR"),
        ([sgpr(4), 7], r"implicit_src\[1\] must be a Register"),
    ],
)
def test_rejects_bad_implicit_sources(implicit, message):
    builder = AsmBuilder("gfx1250", "kernel")
    with pytest.raises(ValueError, match=message):
        builder.instruction("s_wait_idle", [], [], implicit_src=implicit)


def test_rejects_flagged_implicit_source():
    flagged = sgpr(4)
    flagged.set_minus(True)
    with pytest.raises(ValueError, match="negate/absolute"):
        AsmBuilder("gfx1250", "kernel").instruction(
            "s_wait_idle", [], [], implicit_src=[flagged]
        )


def test_rejects_register_offset():
    # A VGPR is named by its full number; StinkyTofu's VGPR-MSB pass derives the offset.
    biased = vgpr(44)
    biased.set_offset(-256)
    with pytest.raises(ValueError, match=r"dst\[0\] has a nonzero Register.offset"):
        AsmBuilder("gfx1250", "kernel").instruction("v_mov_b32", [biased], [vgpr(1)])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda b: b.instruction("not_an_opcode", [], []),
            r"^AsmBuilder\.instruction: unknown instruction 'not_an_opcode' for gfx1250$",
        ),
        (
            lambda b: b.instruction("s_endpgm", [], [], promoted=True),
            "no promoted encoding",
        ),
        (
            lambda b: b.instruction("v_add_nc_u32", [vgpr(0)], [vgpr(1)]),
            "expects 1 dst and 2 src operands",
        ),
        (
            lambda b: b.instruction("v_add_nc_u32", [vgpr(0)], [vgpr(1), sgpr(2)]),
            r"src\[1\] field requires a VGPR",
        ),
        (
            lambda b: b.instruction("v_add_nc_u32", [Register(3)], [vgpr(1), vgpr(2)]),
            "destination must be a physical register",
        ),
        (
            lambda b: b.instruction("v_add_nc_u32", [vgpr(0)], [vgpr(1), 2]),
            r"src\[1\] must be a Register",
        ),
        (lambda b: b.instruction("s_branch", [], []), "requires 'target'"),
        (
            lambda b: b.instruction("s_branch", [], [Register(0)], target="x"),
            "takes 'target', not src operands",
        ),
        (
            lambda b: b.instruction("s_branch", [], [], target="1bad"),
            "not a valid label name",
        ),
        (
            lambda b: b.instruction("s_endpgm", [], [], target="x"),
            "only valid for an instruction",
        ),
        (
            lambda b: b.instruction("s_endpgm", [], [], comment="two\nlines"),
            "single line",
        ),
        (
            lambda b: b.instruction("s_endpgm", [], [], modifiers=[1]),
            "'modifiers' must be a dict",
        ),
        (
            lambda b: b.instruction(
                "ds_load_b128",
                [vgpr(0, 4)],
                [vgpr(4)],
                modifiers={"ds": {"offset0": 1}},
            ),
            "offset0/offset1 require na=2",
        ),
        (
            lambda b: b.instruction("s_endpgm", [], [], modifiers={"ds": {}}),
            "modifier 'ds' is only valid on DS instructions",
        ),
        # Appended to the two-source tensor_load_to_lds form, implicit sources would be read as
        # its third and fourth group, whatever their shape.
        (
            lambda b: b.instruction(
                "tensor_load_to_lds",
                [],
                [sgpr(8, 4), sgpr(12, 8)],
                implicit_src=[sgpr(40, 4), sgpr(44, 4)],
            ),
            "implicit_src is not accepted for 'tensor_load_to_lds' with 2 of its 4 source",
        ),
        (
            lambda b: b.instruction(
                "tensor_load_to_lds",
                [],
                [sgpr(8, 4), sgpr(12, 8)],
                implicit_src=[sgpr(20, 2)],
            ),
            "implicit_src is not accepted for 'tensor_load_to_lds' with 2 of its 4 source",
        ),
        (lambda b: b.label("1bad"), "not a valid label name"),
        (lambda b: b.label("x", 3), "power of two"),
        (lambda b: b.comment("a\nb"), "single line"),
        (lambda b: b.read_write_dests("s_endpgm", True), "no promoted encoding"),
        (
            lambda b: b.read_write_dests("not_an_opcode"),
            r"^AsmBuilder\.read_write_dests: unknown instruction",
        ),
    ],
)
def test_rejected_call_raises_and_builds_nothing(call, message):
    def build(builder, failing):
        builder.label("start")
        builder.instruction("s_mov_b32", [sgpr(1)], [sgpr(2)])
        if failing:
            with pytest.raises(ValueError, match=message):
                call(builder)
        builder.instruction("s_endpgm", [], [])

    assert _built(lambda b: build(b, True)) == _built(lambda b: build(b, False))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    ("mnemonic", "replacement"),
    [
        ("ds_load_b192", "ds_load_b128 and ds_load_b64"),
        ("ds_store_b192", "ds_store_b128 and ds_store_b64"),
        ("ds_store_b256", "two ds_store_b128"),
        ("s_waitcnt", "s_wait_loadcnt, s_wait_storecnt, s_wait_dscnt or s_wait_kmcnt"),
        ("s_barrier", "s_barrier_signal -1 and s_barrier_wait -1"),
    ],
)
def test_rejects_the_forms_the_assembler_rejects(target, mnemonic, replacement):
    # The descriptor table has these, but the assembler does not take them on gfx1250.
    _skip_unbuilt(target)
    expected = f"the assembler rejects '{mnemonic}' on {target}; use {replacement}"

    def build(builder, failing):
        builder.instruction("s_mov_b32", [sgpr(1)], [sgpr(2)])
        if failing:
            with pytest.raises(ValueError) as error:
                builder.instruction(mnemonic, [], [])
            assert str(error.value) == f"AsmBuilder.instruction: {expected}"
        builder.instruction("s_endpgm", [], [])

    assert _built(lambda b: build(b, True), target=target) == _built(
        lambda b: build(b, False), target=target
    )


def test_rejected_label_does_not_define_the_name():
    builder = AsmBuilder("gfx1250", "kernel")
    builder.instruction("s_branch", [], [], target="loop")
    with pytest.raises(ValueError, match="power of two"):
        builder.label("loop", 3)
    with pytest.raises(
        ValueError, match=r"undefined branch target 'loop' \(branch at node 0\)"
    ):
        builder.finish()
    builder.label("loop", 4)  # the rejected call left no duplicate behind
    builder.instruction("s_endpgm", [], [])
    assert ".align 4\nloop:" in builder.finish().emitAssembly()


def test_rejected_branch_leaves_no_target():
    builder = AsmBuilder("gfx1250", "kernel")
    with pytest.raises(ValueError, match="takes 'target'"):
        builder.instruction("s_branch", [], [Register(0)], target="nowhere")
    builder.instruction("s_endpgm", [], [])
    assert "s_endpgm" in builder.finish().emitAssembly()


def test_duplicate_label_is_rejected():
    builder = AsmBuilder("gfx1250", "kernel")
    builder.label("x")
    with pytest.raises(ValueError, match="duplicate label 'x'"):
        builder.label("x")


def test_finish_closes_the_builder_and_the_module_outlives_it():
    builder = AsmBuilder("gfx1250", "kernel")
    builder.instruction("s_endpgm", [], [])
    module = builder.finish()
    for call in (
        lambda: builder.instruction("s_endpgm", [], []),
        lambda: builder.label("x"),
        lambda: builder.comment("x"),
        builder.blank,
        builder.finish,
    ):
        with pytest.raises(ValueError, match="finish\\(\\) was already called"):
            call()
    del builder
    gc.collect()
    assert "s_endpgm" in module.emitAssembly()


def test_finish_checks_read_write_sources_only_when_legalized():
    # Without legalize the read of s5 is not added, and finish() does not ask for it.
    builder = AsmBuilder("gfx1250", "kernel")
    builder.instruction("s_addk_co_i32", [sgpr(5)], [Register(1)], legalize=False)
    builder.instruction("s_endpgm", [], [])
    assert "s_addk_co_i32 s5, 1" in _lines(builder.finish().emitAssembly())


@pytest.mark.parametrize(
    ("mnemonic", "promoted", "positions"),
    [
        ("s_addk_co_i32", False, [0]),
        ("s_bitset0_b32", False, [0]),
        ("s_bitset1_b32", False, [0]),
        ("s_cmov_b32", False, [0]),
        ("v_swap_b32", False, [0, 1]),
        ("v_add_f32", False, []),
        ("v_add_f32", True, []),
        ("v_cvt_pk_fp8_f32", False, []),  # read-write only by an instance's op_sel
    ],
)
def test_read_write_dests_reads_the_descriptor(mnemonic, promoted, positions):
    assert (
        AsmBuilder("gfx1250", "kernel").read_write_dests(mnemonic, promoted)
        == positions
    )


def _loop(builder):
    builder.instruction("s_mov_b32", [sgpr(0)], [Register(0)])
    builder.label("loop_top")
    builder.instruction("s_add_u32", [sgpr(0)], [sgpr(0), Register(1)])
    builder.comment("tail")
    builder.instruction("s_cmp_lt_u32", [], [sgpr(0), sgpr(1)])
    builder.instruction("s_cbranch_scc1", [], [], target="loop_top")
    builder.blank()
    builder.label("loop_exit")
    builder.instruction("s_endpgm", [], [])


def test_set_group_range_covers_the_loop_by_label():
    builder = AsmBuilder("gfx1250", "kernel")
    _loop(builder)
    info = builder.set_group_range("loopWithPrefetch", "loop_top", "loop_exit")
    assert info == {
        "group": "loopWithPrefetch",
        "first": "loop_top",
        "last_mnemonic": "s_cbranch_scc1",
        "nodes": 5,
        "instructions": 3,
        "labels": 1,
        "directives": 1,
    }
    inclusive = builder.set_group_range(
        "loopWithPrefetch", "loop_top", "loop_exit", end_inclusive=True
    )
    assert inclusive["last_label"] == "loop_exit" and inclusive["nodes"] == 7


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("noSuchGroup", "loop_top", "loop_exit"), "registered no group 'noSuchGroup'"),
        (
            ("loopWithPrefetch", "missing", "loop_exit"),
            "label 'missing' is not in the entry block \\(no such label was built\\)",
        ),
        (
            ("loopWithPrefetch", "loop_exit", "loop_top"),
            "end_label does not follow first_label",
        ),
        (
            ("loopWithPrefetch", "loop_top", "loop_top"),
            "the range holds no instruction besides labels",
        ),
    ],
)
def test_set_group_range_rejects_bad_ranges(args, message):
    builder = AsmBuilder("gfx1250", "kernel")
    _loop(builder)
    builder.finish()  # allowed after finish()
    with pytest.raises(ValueError, match=message):
        builder.set_group_range(*args)


def test_set_group_range_rejects_a_range_of_labels_only():
    builder = AsmBuilder("gfx1250", "kernel")
    builder.label("first")
    builder.comment("between")
    builder.blank()
    builder.label("second", 16)
    builder.label("third")
    builder.instruction("s_endpgm", [], [])
    for args, inclusive in (
        (("first", "second"), False),
        (("first", "third"), False),
        (("first", "third"), True),
        (("first", "first"), True),
    ):
        with pytest.raises(
            ValueError, match="the range holds no instruction besides labels"
        ):
            builder.set_group_range("loopWithPrefetch", *args, end_inclusive=inclusive)


@pytest.mark.parametrize("allocate", [False, True], ids=["moved", "deleted"])
def test_set_group_range_after_the_pipeline_raises(allocate):
    # The pipeline moves labels out of the entry block, and with register allocation on it
    # deletes the unreachable block after s_endpgm together with its labels.
    options = {"OptLevel": 0}
    if allocate:
        options.update(
            {
                "RegisterAllocation": 2,
                "RegisterAllocationConfig": "classes=v,allocator=greedy",
            }
        )
    builder = AsmBuilder("gfx1250", "kernel", options)
    _loop(builder)
    builder.label("dead_top")
    builder.instruction("s_add_u32", [sgpr(2)], [sgpr(2), Register(1)])
    builder.label("dead_end")
    builder.instruction("s_endpgm", [], [])
    module = builder.finish()
    module.runOptimizationPipeline()
    # Only register allocation deletes the unreachable block.
    assert ("dead_top:" in module.emitAssembly()) == (not allocate)
    for first, end in (("loop_top", "loop_exit"), ("dead_top", "dead_end")):
        with pytest.raises(
            ValueError, match="runOptimizationPipeline\\(\\) already ran"
        ):
            builder.set_group_range("loopWithPrefetch", first, end)


@pytest.mark.parametrize(
    ("arch", "options", "message"),
    [
        (
            "gfx9999",
            None,
            r"^AsmBuilder\.__init__: unsupported architecture 'gfx9999'$",
        ),
        ("gfx1250", [], "'options' must be a dict or None"),
        ("gfx1250", {"EnableESM": True}, "unknown option 'EnableESM'"),
        ("gfx1250", {1: True}, "option keys must be strings"),
        ("gfx1250", {"OptLevel": True}, "option 'OptLevel' has the wrong type"),
        ("gfx1250", {"OptLevel": "1"}, "option 'OptLevel' has the wrong type"),
        ("gfx1250", {"EnableESM2": 1}, "option 'EnableESM2' has the wrong type"),
        ("gfx1250", {"KeepAuthored": 1}, "option 'KeepAuthored' has the wrong type"),
        (
            "gfx1250",
            {"KeepAuthored": "nop,MOV"},
            r"KeepAuthored: unknown token 'MOV' \(known tokens: nop, mov, reuse\)",
        ),
        ("gfx1250", {"ArchName": "gfx950"}, "option 'ArchName' names 'gfx950'"),
        ("gfx1250", {"wavefrontSize": 64}, "option 'wavefrontSize' is 64"),
    ],
)
def test_constructor_rejects_bad_arguments(arch, options, message):
    with pytest.raises(ValueError, match=message):
        AsmBuilder(arch, "kernel", options)


@pytest.mark.parametrize("target", TARGETS)
def test_constructor_accepts_matching_target_options(target):
    _skip_unbuilt(target)
    AsmBuilder(
        target,
        "kernel",
        {
            "ArchName": target,
            "wavefrontSize": 32,
            "KeepAuthored": " nop , reuse,",
            "OptLevel": 0,
        },
    )


def _keep_authored_kernel(builder):
    builder.instruction("s_mov_b32", [sgpr(21)], [sgpr(23)])
    builder.instruction("v_nop", [], [])
    builder.instruction("s_add_u32", [sgpr(30)], [sgpr(21), sgpr(24)])
    builder.instruction("s_mov_b32", [sgpr(21)], [sgpr(26)])
    builder.instruction(
        "v_wmma_f32_16x16x32_bf16",
        [vgpr(64, 8)],
        [vgpr(72, 8), vgpr(80, 8), vgpr(64, 8)],
    )
    builder.instruction(
        "v_wmma_f32_16x16x32_bf16",
        [vgpr(88, 8)],
        [vgpr(96, 8), vgpr(80, 8), vgpr(88, 8)],
    )
    builder.instruction("s_endpgm", [], [])


def _pipeline_options(opt_level):
    options = {"OptLevel": opt_level}
    if opt_level > 0:
        # The backend entry needs a tile configuration from -O1 on; its values do not matter.
        options.update({"TileA0": 1, "TileB0": 1, "TileM0": 1})
    return options


def _pipeline_output(keep, opt_level=0):
    options = _pipeline_options(opt_level)
    if keep is not None:
        options["KeepAuthored"] = keep
    builder = AsmBuilder("gfx1250", "kernel", options)
    _keep_authored_kernel(builder)
    module = builder.finish()
    module.runOptimizationPipeline()
    return _lines(module.emitAssembly())


@pytest.mark.parametrize("opt_level", [0, 1])
def test_pipeline_rewrites_authored_code_without_keep_authored(opt_level):
    lines = _pipeline_output(None, opt_level)
    assert lines == _pipeline_output("", opt_level)
    assert "s_add_u32 s30, s23, s24" in lines  # the copy was propagated
    assert "s_mov_b32 s21, s23" not in lines  # and the dead mov erased
    assert lines.count("v_nop") == 1  # only the entry prologue's own v_nop
    assert any("matrix_b_reuse" in line for line in lines)


@pytest.mark.parametrize("opt_level", [0, 1])
def test_keep_authored_leaves_each_part_as_written(opt_level):
    unset = _pipeline_output(None, opt_level)
    nop = _pipeline_output("nop", opt_level)
    assert nop.count("v_nop") == unset.count("v_nop") + 1
    assert "s_add_u32 s30, s23, s24" in nop

    mov = _pipeline_output("mov", opt_level)
    assert "s_mov_b32 s21, s23" in mov and "s_add_u32 s30, s21, s24" in mov
    assert mov.count("v_nop") == unset.count("v_nop")

    reuse = _pipeline_output("reuse", opt_level)
    assert not any("matrix_b_reuse" in line for line in reuse)
    assert "s_add_u32 s30, s23, s24" in reuse

    every = _pipeline_output("nop,mov,reuse", opt_level)
    assert every.count("v_nop") == nop.count("v_nop")
    assert "s_add_u32 s30, s21, s24" in every
    assert not any("matrix_b_reuse" in line for line in every)


def test_keep_authored_nop_also_covers_the_wait_insertion_region():
    def output(keep):
        options = {"OptLevel": 0, "EnableWaitCntInsertion": True}
        if keep:
            options["KeepAuthored"] = keep
        builder = AsmBuilder("gfx1250", "kernel", options)
        builder.instruction("s_mov_b32", [sgpr(0)], [Register(0)])
        builder.label("loop_top")
        builder.instruction("s_nop", [], [Register(0)])
        builder.instruction("s_add_u32", [sgpr(0)], [sgpr(0), Register(1)])
        builder.instruction("s_cmp_lt_u32", [], [sgpr(0), sgpr(1)])
        builder.instruction("s_cbranch_scc1", [], [], target="loop_top")
        builder.label("loop_exit")
        builder.instruction("s_endpgm", [], [])
        builder.set_group_range("loopWithPrefetch", "loop_top", "loop_exit")
        module = builder.finish()
        module.runOptimizationPipeline()
        return _lines(module.emitAssembly())

    assert output("nop").count("s_nop 0") == output(None).count("s_nop 0") + 1
