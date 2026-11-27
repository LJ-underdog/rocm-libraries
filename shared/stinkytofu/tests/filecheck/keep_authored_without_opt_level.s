# RUN: %stinkytofu-opt --arch gfx1250 %s --keep-authored=nop --emit-asm 2>&1
# XFAIL
#
# KeepAuthored gates passes of the registered pipeline, so stinkytofu-opt takes
# --keep-authored only together with -O<N>. Without it the tool reports this error and exits
# with status 1, which XFAIL notes but does not check.
#
# CHECK: Error: --keep-authored applies to the -O<N> pipeline only

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

keep_authored:
    s_endpgm
