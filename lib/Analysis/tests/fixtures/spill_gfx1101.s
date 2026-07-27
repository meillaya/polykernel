	.amdgcn_target "amdgcn-amd-amdhsa--gfx1101"
	.amdhsa_code_object_version 6
	.section	.text._Z10spill_testv,"axG",@progbits,_Z10spill_testv,comdat
	.globl	_Z10spill_testv ; -- Begin function _Z10spill_testv
	.p2align	8
	.type	_Z10spill_testv,@function
_Z10spill_testv: ; @_Z10spill_testv
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x18
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0x3f800000
	v_mov_b32_e32 v2, 0x40000000
	v_mov_b32_e32 v3, 0x40400000
	v_mov_b32_e32 v4, 0x40800000
	v_mov_b32_e32 v5, 0x40a00000
	v_mov_b32_e32 v6, 0x40c00000
	v_mov_b32_e32 v7, 0x40e00000
	v_mov_b32_e32 v8, 0x41000000
	v_mov_b32_e32 v9, 0x41100000
	v_mov_b32_e32 v10, 0x41200000
	v_mov_b32_e32 v11, 0x41300000
	v_mov_b32_e32 v12, 0x41400000
	v_mov_b32_e32 v13, 0x41500000
	v_mov_b32_e32 v14, 0x41600000
	v_mov_b32_e32 v15, 0x41700000
	v_mov_b32_e32 v16, 0x41800000
	v_mov_b32_e32 v17, 0x41880000
	v_mov_b32_e32 v18, 0x41900000
	v_mov_b32_e32 v19, 0x41980000
	; Register pressure forces spills to scratch (private segment):
	scratch_store_b32 v19, off offset:0
	scratch_store_b32 v18, off offset:4
	scratch_store_b32 v17, off offset:8
	scratch_store_b32 v16, off offset:12
	s_waitcnt vmcnt(0)
	scratch_load_b32 v16, off offset:12
	scratch_load_b32 v17, off offset:8
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v0, v16, v17
	v_add_f32_e32 v0, v0, v1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10spill_testv
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 16
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 20
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
	.section	.text._Z10spill_testv,"axG",@progbits,_Z10spill_testv,comdat
.Lfunc_end0:
	.size	_Z10spill_testv, .Lfunc_end0-_Z10spill_testv
                                        ; -- End function
	.set _Z10spill_testv.num_vgpr, 20
	.set _Z10spill_testv.num_agpr, 0
	.set _Z10spill_testv.numbered_sgpr, 16
	.set _Z10spill_testv.num_named_barrier, 0
	.set _Z10spill_testv.private_seg_size, 16
	.set _Z10spill_testv.uses_vcc, 1
	.set _Z10spill_testv.uses_flat_scratch, 0
	.set _Z10spill_testv.has_dyn_sized_stack, 0
	.set _Z10spill_testv.has_recursion, 0
	.set _Z10spill_testv.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 512
; TotalNumSgprs: 18
; NumVgprs: 20
; ScratchSize: 16
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 20
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
	.ident	"nixpkgs-AMD clang version 22.0.0 (https://github.com/ROCm/llvm-project/tree/rocm-7.2.3 rocm-7.2.3)"
	.section	".note.GNU-stack","",@progbits
