# RUN: %stinkytofu-opt --arch gfx1250 %s -O0 --keep-authored=nop,mov,reuse --emit-asm
#
# --keep-authored sets ModuleOptions.KeepAuthored: the -O0 pipeline then runs neither
# StinkyRemoveNopPass, AsmMovePropagationPass nor SetMatrixReusePass, so the authored v_nop,
# the s_mov_b32 seeding s21 and its read stay as written, and no matrix-reuse hint is added.
# keep_authored_unset_o0.s runs the same input without the option.
#
# CHECK: global_prefetch_b8
# CHECK: s_mov_b32 s21, s23
# CHECK-NEXT: v_nop
# CHECK-NEXT: s_add_u32 s30, s21, s24
# CHECK-NEXT: s_mov_b32 s21, s26
# CHECK-NOT: matrix_b_reuse
# CHECK: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

keep_authored:
    s_mov_b32 s21, s23
    v_nop
    s_add_u32 s30, s21, s24
    s_mov_b32 s21, s26
    v_wmma_f32_16x16x32_bf16 v[64:71], v[72:79], v[80:87], v[64:71]
    v_wmma_f32_16x16x32_bf16 v[88:95], v[96:103], v[80:87], v[88:95]
    s_endpgm
