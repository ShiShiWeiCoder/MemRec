#!/usr/bin/env python3
"""P0 消融: MultilevelMemoryHEA 三种 fusion_mode 静态验证
不占 GPU, 只做 CPU forward 形状检查.
用法: python ablation_experiments/tests/test_p0_fusion_modes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from RS.layers import MultilevelMemoryHEA


def make_memory_data(batch_size, transition_dim=9, enhanced_gating=True):
    return {
        'sensory_memory_ratio': torch.full((batch_size,), 0.3),
        'working_memory_ratio': torch.full((batch_size,), 0.4),
        'longterm_memory_ratio': torch.full((batch_size,), 0.3),
        'enhanced_gating_features': torch.randn(batch_size, 6) if enhanced_gating else None,
        'transition_features': torch.randn(batch_size, transition_dim) if transition_dim > 0 else None,
    }


def run_case(fusion_mode, inp_dim=32, analysis_dim=64, batch_size=4):
    print(f"\n===== fusion_mode={fusion_mode}  inp_dim={inp_dim} analysis_dim={analysis_dim} =====")
    ple_arch = [2, 3, [128, inp_dim], 1]
    hea = MultilevelMemoryHEA(
        ple_arch, inp_dim, dropout=0.1,
        enhanced_gating=True, reflection_mode=True, transition_feature_dim=9,
        analysis_dim=analysis_dim, fusion_mode=fusion_mode,
    )
    vn_user = torch.randn(batch_size, inp_dim)
    vn_course = torch.randn(batch_size, inp_dim)
    vn_analysis = torch.randn(batch_size, analysis_dim)
    mem = make_memory_data(batch_size, transition_dim=9, enhanced_gating=True)
    mem = {k: v for k, v in mem.items() if v is not None}
    out = hea([vn_user, vn_course, vn_analysis], multilevel_memory_data=mem)
    if isinstance(out, (list, tuple)):
        for i, t in enumerate(out):
            print(f"  out[{i}].shape = {tuple(t.shape)}")
    else:
        print(f"  out.shape = {tuple(out.shape)}")
    loss = sum(t.pow(2).mean() for t in (out if isinstance(out, (list, tuple)) else [out]))
    loss.backward()
    n_params = sum(p.numel() for p in hea.parameters())
    n_grads = sum(p.numel() for p in hea.parameters() if p.grad is not None)
    print(f"  params={n_params}  grads_computed={n_grads}  backward OK")
    return True


if __name__ == '__main__':
    torch.manual_seed(0)
    for mode in ['film', 'concat', 'xattn']:
        for dims in [(32, 64), (64, 64), (768, 768)]:
            run_case(mode, inp_dim=dims[0], analysis_dim=dims[1])
    print("\n[OK] 所有 fusion_mode 均通过静态验证")
