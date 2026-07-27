	.amdgcn_target "amdgcn-amd-amdhsa--gfx1101"
	.amdhsa_code_object_version 6
	.section	.text._ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii,"axG",@progbits,_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii,comdat
	.globl	_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii ; -- Begin function _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii
	.p2align	8
	.type	_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii,@function
_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii: ; @_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x18
	s_load_b64 s[8:9], s[0:1], 0x10
	v_and_b32_e32 v9, 0x3ff, v0
	v_bfe_u32 v8, v0, 10, 10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v1, s2, 4, v9
	v_lshl_add_u32 v7, s3, 4, v8
	s_delay_alu instid0(VALU_DEP_2)
	v_ashrrev_i32_e32 v2, 31, v1
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e64 s2, s5, v1
	s_cmp_lt_i32 s6, 1
	v_cmp_gt_i32_e32 vcc_lo, s4, v7
	s_cbranch_scc1 .LBB0_8
; %bb.1:                                ; %.lr.ph
	v_mad_u64_u32 v[3:4], null, v7, s6, 0
	v_ashrrev_i32_e32 v6, 31, v7
	s_load_b128 s[12:15], s[0:1], 0x0
	v_lshlrev_b64 v[15:16], 1, v[1:2]
	s_add_i32 s1, s6, 15
	s_ashr_i32 s11, s5, 31
	s_mov_b32 s10, s5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v0, v4 :: v_dual_lshlrev_b32 v11, 5, v8
	s_lshr_b32 s1, s1, 4
	s_lshl_b64 s[10:11], s[10:11], 5
	v_mad_u64_u32 v[4:5], null, v6, s6, v[0:1]
	v_mad_i64_i32 v[5:6], null, s5, v8, 0
	v_lshlrev_b32_e32 v0, 1, v9
	v_mov_b32_e32 v10, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	v_add_nc_u32_e32 v12, 0x200, v0
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_add_nc_u32_e32 v13, v11, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v14, v12, v11
	v_add_co_u32 v0, s0, v3, v0
	v_add_co_ci_u32_e64 v4, null, 0, v4, s0
	v_add_co_u32 v5, s0, v5, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, v6, v16, s0
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v3, s0, s12, v0
	v_add_co_ci_u32_e64 v4, null, s13, v4, s0
	v_add_co_u32 v5, s0, s14, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s15, v6, s0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt vmcnt(0)
	ds_store_b16 v14, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_b128 v[15:18], v11
	ds_load_u16 v0, v12
	ds_load_u16 v23, v12 offset:32
	ds_load_u16 v24, v12 offset:64
	ds_load_b128 v[19:22], v11 offset:16
	ds_load_u16 v25, v12 offset:96
	ds_load_u16 v26, v12 offset:128
	ds_load_u16 v27, v12 offset:160
	ds_load_u16 v28, v12 offset:192
	ds_load_u16 v29, v12 offset:224
	v_add_co_u32 v3, s0, v3, 32
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, 0, v4, s0
	v_add_co_u32 v5, s0, v5, s10
	v_add_co_ci_u32_e64 v6, null, s11, v6, s0
	v_add_nc_u32_e32 v8, 16, v8
	s_add_i32 s1, s1, -1
	s_waitcnt lgkmcnt(9)
	v_lshlrev_b32_e32 v30, 16, v15
	s_waitcnt lgkmcnt(8)
	v_lshlrev_b32_e32 v0, 16, v0
	v_and_b32_e32 v15, 0xffff0000, v15
	s_waitcnt lgkmcnt(6)
	v_lshlrev_b32_e32 v24, 16, v24
	s_cmp_eq_u32 s1, 0
	v_fmac_f32_e32 v10, v30, v0
	v_lshlrev_b32_e32 v0, 16, v16
	v_lshlrev_b32_e32 v23, 16, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v10, v15, v23 :: v_dual_and_b32 v15, 0xffff0000, v16
	s_waitcnt lgkmcnt(4)
	v_lshlrev_b32_e32 v16, 16, v25
	v_fmac_f32_e32 v10, v0, v24
	v_lshlrev_b32_e32 v0, 16, v17
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b32_e32 v23, 16, v26
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v24, 16, v28
	v_lshlrev_b32_e32 v28, 16, v19
	v_fmac_f32_e32 v10, v15, v16
	ds_load_u16 v15, v12 offset:256
	v_and_b32_e32 v16, 0xffff0000, v17
	v_dual_fmac_f32 v10, v0, v23 :: v_dual_lshlrev_b32 v23, 16, v18
	v_lshlrev_b32_e32 v17, 16, v27
	ds_load_u16 v0, v12 offset:288
	v_dual_fmac_f32 v10, v16, v17 :: v_dual_and_b32 v17, 0xffff0000, v18
	ds_load_u16 v16, v12 offset:320
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b32_e32 v18, 16, v29
	v_fmac_f32_e32 v10, v23, v24
	ds_load_u16 v23, v12 offset:352
	ds_load_u16 v24, v12 offset:384
	ds_load_u16 v25, v12 offset:416
	ds_load_u16 v26, v12 offset:448
	ds_load_u16 v27, v12 offset:480
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_dual_fmac_f32 v10, v17, v18 :: v_dual_lshlrev_b32 v15, 16, v15
	buffer_gl0_inv
	v_dual_fmac_f32 v10, v28, v15 :: v_dual_lshlrev_b32 v15, 16, v20
	v_lshlrev_b32_e32 v0, 16, v0
	v_lshlrev_b32_e32 v16, 16, v16
	v_and_b32_e32 v17, 0xffff0000, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v10, v17, v0 :: v_dual_lshlrev_b32 v17, 16, v23
	v_and_b32_e32 v0, 0xffff0000, v20
	v_dual_fmac_f32 v10, v15, v16 :: v_dual_lshlrev_b32 v15, 16, v21
	v_lshlrev_b32_e32 v16, 16, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v10, v0, v17
	v_and_b32_e32 v0, 0xffff0000, v21
	v_dual_fmac_f32 v10, v15, v16 :: v_dual_lshlrev_b32 v17, 16, v25
	v_lshlrev_b32_e32 v15, 16, v22
	v_lshlrev_b32_e32 v16, 16, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v10, v0, v17 :: v_dual_lshlrev_b32 v17, 16, v27
	v_add_nc_u32_e32 v9, 16, v9
	v_and_b32_e32 v0, 0xffff0000, v22
	v_fmac_f32_e32 v10, v15, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v10, v0, v17
	s_cbranch_scc1 .LBB0_7
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b16_e32 v0.l, 0
	v_cmp_gt_i32_e64 s0, s6, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mov_b16_e32 v0.h, v0.l
	s_and_b32 s3, vcc_lo, s0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_d16_hi_b16 v0, v[3:4], off
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_cmp_gt_i32_e64 s0, s6, v8
	s_waitcnt vmcnt(0)
	ds_store_b16_d16_hi v13, v0
	s_and_b32 s3, s2, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB0_2
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_d16_b16 v0, v[5:6], off
	s_branch .LBB0_2
.LBB0_7:                                ; %._crit_edge.loopexit
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v0, v10, 16, 1
	v_or_b32_e32 v3, 0x400000, v10
	v_cmp_u_f32_e32 vcc_lo, v10, v10
	v_add3_u32 v0, v0, v10, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, v0, v3, vcc_lo
	s_branch .LBB0_9
.LBB0_8:
	v_mov_b16_e32 v0.h, 0
.LBB0_9:                                ; %Flow
	v_cmp_gt_i32_e32 vcc_lo, s4, v7
	v_cmp_gt_i32_e64 s0, s5, v1
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_11
; %bb.10:
	v_mad_i64_i32 v[3:4], null, s5, v7, 0
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[3:4], 1, v[3:4]
	v_add_co_u32 v3, vcc_lo, s8, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v4, vcc_lo
	v_add_co_u32 v1, vcc_lo, v3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, v4, v2, vcc_lo
	global_store_d16_hi_b16 v[1:2], v0, off
.LBB0_11:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii
		.amdhsa_group_segment_fixed_size 1024
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 31
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 9
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii,"axG",@progbits,_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii,comdat
.Lfunc_end0:
	.size	_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii, .Lfunc_end0-_ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii
                                        ; -- End function
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.num_vgpr, 31
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.num_agpr, 0
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.numbered_sgpr, 16
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.num_named_barrier, 0
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.private_seg_size, 0
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.uses_vcc, 1
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.uses_flat_scratch, 0
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.has_dyn_sized_stack, 0
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.has_recursion, 0
	.set _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1064
; TotalNumSgprs: 18
; NumVgprs: 31
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 31
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_7c3bb562c5cc4fdb,@object ; @__hip_cuid_7c3bb562c5cc4fdb
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7c3bb562c5cc4fdb
__hip_cuid_7c3bb562c5cc4fdb:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7c3bb562c5cc4fdb, 1

	.ident	"nixpkgs-AMD clang version 22.0.0 (https://github.com/ROCm/llvm-project/tree/rocm-7.2.3 rocm-7.2.3)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7c3bb562c5cc4fdb
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _ZN12_GLOBAL__N_113matmul_kernelEPK14__hip_bfloat16S2_PS0_iii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     31
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1101
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
