# RUN: %stinkytofu-opt --arch gfx1250 %s -O0 --keep-authored=nop,MOV --emit-asm 2>&1
# XFAIL
#
# KeepAuthored tokens are lower-case and matched exactly. stinkytofu-opt reports an unknown
# token with this message before it builds the pipeline (whose own report would start with
# "FATAL ERROR") and exits with status 1, which XFAIL notes but does not check.
#
# CHECK: Error: KeepAuthored: unknown token 'MOV' (known tokens: nop, mov, reuse)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

keep_authored:
    s_endpgm
