# RUN: %stinkytofu-opt --arch gfx1250 %s -O0 --emit-asm
#
# The input of keep_authored_o0.s without --keep-authored: StinkyRemoveNopPass removes the
# v_nop, AsmMovePropagationPass rewrites the read of s21 to s23 and erases the dead
# s_mov_b32, and SetMatrixReusePass marks the shared B matrix for reuse. The only v_nop left
# is the one InsertInitialUnclausedVmemPass puts before the global_prefetch_b8.
#
# CHECK: v_nop
# CHECK-NEXT: global_prefetch_b8
# CHECK-NOT: s_mov_b32 s21, s23
# CHECK-NOT: v_nop
# CHECK: s_add_u32 s30, s23, s24
# CHECK-NEXT: s_mov_b32 s21, s26
# CHECK: matrix_b_reuse
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
