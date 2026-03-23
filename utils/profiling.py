#!/usr/bin/env python3
"""
Profiling utilities for MoE Geometric Classifier
Measures training time, model size, FLOPs, and memory usage
"""

import time
import torch
import json
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")


class MoEProfiler:
    """
    Comprehensive profiler for MoE models

    Tracks:
    - Model size (parameters, checkpoint file size)
    - FLOPs (gate network, expert networks, total)
    - Training time (epoch, data loading, compute)
    - Memory usage (peak GPU, activation memory)
    """

    def __init__(self, model, device='cuda', save_dir='./profiling'):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for metrics
        self.metrics = {
            'model_info': {},
            'flops': {},
            'memory': {},
            'timing': {},
        }

        # Timing state
        self.epoch_start_time = None
        self.batch_times = []
        self.data_times = []
        self.compute_times = []

    def profile_model_size(self) -> Dict:
        """Profile model parameters and checkpoint size"""

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        # MoE-specific: count gate and expert parameters
        gate_params = sum(p.numel() for p in self.model.gate.parameters())
        expert_params = sum(p.numel() for p in self.model.experts.parameters())
        params_per_expert = expert_params // self.model.num_experts

        # Model size in memory (float32 = 4 bytes)
        model_size_mb = (total_params * 4) / (1024 ** 2)

        # Save checkpoint to measure actual size
        checkpoint_path = self.save_dir / 'temp_checkpoint.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
        checkpoint_path.unlink()  # Delete temp file

        info = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'non_trainable_params': int(non_trainable_params),
            'gate_params': int(gate_params),
            'expert_params': int(expert_params),
            'params_per_expert': int(params_per_expert),
            'num_experts': self.model.num_experts,
            'model_size_mb': float(model_size_mb),
            'checkpoint_size_mb': float(checkpoint_size_mb),
        }

        self.metrics['model_info'] = info
        return info

    def profile_flops(self, input_shape: Tuple[int, ...],
                     batch_size: int = 1) -> Dict:
        """
        Profile FLOPs for MoE model

        Args:
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Batch size for profiling
        """

        if not THOP_AVAILABLE:
            print("Warning: thop not available. Returning manual estimates.")
            return self._manual_flop_estimation(input_shape, batch_size)

        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

        # Profile gate network
        try:
            gate_macs, gate_params = profile(
                self.model.gate,
                inputs=(dummy_input,),
                verbose=False
            )
        except Exception as e:
            print(f"Gate profiling failed: {e}")
            gate_macs = 0

        # Profile single expert
        try:
            expert_macs, expert_params = profile(
                self.model.experts[0],
                inputs=(dummy_input,),
                verbose=False
            )
        except Exception as e:
            print(f"Expert profiling failed: {e}")
            expert_macs = 0

        # Total MACs (soft routing: all experts computed)
        total_macs = gate_macs + (expert_macs * self.model.num_experts)

        # Convert MACs to FLOPs (1 MAC ≈ 2 FLOPs)
        gate_flops = gate_macs * 2
        expert_flops = expert_macs * 2
        total_flops = total_macs * 2

        # Per-sample metrics
        gate_flops_per_sample = gate_flops / batch_size
        expert_flops_per_sample = expert_flops / batch_size
        total_flops_per_sample = total_flops / batch_size

        flops_info = {
            'gate_macs': int(gate_macs),
            'gate_flops': int(gate_flops),
            'expert_macs': int(expert_macs),
            'expert_flops': int(expert_flops),
            'total_macs': int(total_macs),
            'total_flops': int(total_flops),
            'gate_gflops': float(gate_flops / 1e9),
            'expert_gflops': float(expert_flops / 1e9),
            'total_gflops': float(total_flops / 1e9),
            'flops_per_sample': int(total_flops_per_sample),
            'gflops_per_sample': float(total_flops_per_sample / 1e9),
            'batch_size': batch_size,
        }

        self.metrics['flops'] = flops_info
        return flops_info

    def _manual_flop_estimation(self, input_shape, batch_size) -> Dict:
        """Manual FLOPs estimation when thop is not available"""

        input_dim = input_shape[0]

        def linear_flops(in_dim, out_dim, bs):
            return bs * (2 * in_dim * out_dim + out_dim)

        # Gate network FLOPs
        gate_hidden = getattr(self.model, 'gate_hidden', 128)
        gate_flops = (
            linear_flops(input_dim, gate_hidden, batch_size) +
            linear_flops(gate_hidden, 64, batch_size) +
            linear_flops(64, self.model.num_experts, batch_size)
        )

        # Expert network FLOPs
        hidden_dim = getattr(self.model, 'hidden_dim', 256)
        expert_flops = (
            linear_flops(input_dim, hidden_dim, batch_size) +
            linear_flops(hidden_dim, 128, batch_size) +
            linear_flops(128, 3, batch_size)
        )

        total_flops = gate_flops + (expert_flops * self.model.num_experts)

        return {
            'gate_flops': int(gate_flops),
            'expert_flops': int(expert_flops),
            'total_flops': int(total_flops),
            'gate_gflops': float(gate_flops / 1e9),
            'expert_gflops': float(expert_flops / 1e9),
            'total_gflops': float(total_flops / 1e9),
            'method': 'manual_estimation'
        }

    def profile_memory(self, input_tensor: torch.Tensor) -> Dict:
        """
        Profile GPU memory usage

        Args:
            input_tensor: Sample input tensor for forward+backward pass
        """

        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()

        # Measure model parameter memory
        model_mem = sum([p.numel() * p.element_size()
                        for p in self.model.parameters()])
        model_mem_mb = model_mem / (1024 ** 2)

        # Forward pass
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated(self.device)

        # Forward + backward (simulated training)
        self.model.train()
        torch.cuda.synchronize()

        output = self.model(input_tensor)
        loss = output.sum()  # Dummy loss
        loss.backward()

        torch.cuda.synchronize()
        mem_after_backward = torch.cuda.memory_allocated(self.device)

        # Peak memory
        peak_mem = torch.cuda.max_memory_allocated(self.device)

        memory_info = {
            'model_params_mb': float(model_mem_mb),
            'forward_activations_mb': float((mem_after_forward - mem_before) / (1024 ** 2)),
            'backward_gradients_mb': float((mem_after_backward - mem_after_forward) / (1024 ** 2)),
            'peak_memory_mb': float(peak_mem / (1024 ** 2)),
            'total_allocated_mb': float(mem_after_backward / (1024 ** 2)),
        }

        self.metrics['memory'] = memory_info
        return memory_info

    def start_epoch(self):
        """Start epoch timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_start_time = time.perf_counter()
        self.batch_times = []
        self.data_times = []
        self.compute_times = []

    def record_batch_timing(self, data_time: float, compute_time: float):
        """
        Record timing for a single batch

        Args:
            data_time: Time spent loading data (seconds)
            compute_time: Time spent on forward+backward+optimizer (seconds)
        """
        self.data_times.append(data_time)
        self.compute_times.append(compute_time)
        self.batch_times.append(data_time + compute_time)

    def end_epoch(self, num_samples: int) -> Dict:
        """
        End epoch timing and compute statistics

        Args:
            num_samples: Total number of samples in epoch
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - self.epoch_start_time

        total_data_time = sum(self.data_times)
        total_compute_time = sum(self.compute_times)

        timing_info = {
            'epoch_time': float(epoch_time),
            'data_time': float(total_data_time),
            'compute_time': float(total_compute_time),
            'data_fraction': float(total_data_time / epoch_time) if epoch_time > 0 else 0.0,
            'compute_fraction': float(total_compute_time / epoch_time) if epoch_time > 0 else 0.0,
            'samples_per_second': float(num_samples / epoch_time) if epoch_time > 0 else 0.0,
            'avg_batch_time': float(sum(self.batch_times) / len(self.batch_times)) if self.batch_times else 0.0,
        }

        # Update metrics (keep latest)
        self.metrics['timing'] = timing_info
        return timing_info

    def get_summary(self) -> Dict:
        """Get comprehensive profiling summary"""
        return self.metrics

    def print_summary(self):
        """Print formatted profiling summary"""

        print("\n" + "="*80)
        print("MoE Model Profiling Summary")
        print("="*80)

        # Model info
        if 'model_info' in self.metrics and self.metrics['model_info']:
            info = self.metrics['model_info']
            print("\n[MODEL] Model Architecture:")
            print(f"  Total parameters: {info['total_params']:,}")
            print(f"    - Trainable: {info['trainable_params']:,}")
            print(f"    - Non-trainable: {info['non_trainable_params']:,}")
            print(f"  Gate network: {info['gate_params']:,} params")
            print(f"  Expert networks: {info['expert_params']:,} params total")
            print(f"    - Per expert: {info['params_per_expert']:,} params")
            print(f"    - Num experts: {info['num_experts']}")
            print(f"  Model size (float32): {info['model_size_mb']:.2f} MB")
            print(f"  Checkpoint size: {info['checkpoint_size_mb']:.2f} MB")

        # FLOPs
        if 'flops' in self.metrics and self.metrics['flops']:
            flops = self.metrics['flops']
            print("\n[FLOPS] Computational Complexity:")
            print(f"  Gate network: {flops.get('gate_gflops', 0):.4f} GFLOPs")
            print(f"  Single expert: {flops.get('expert_gflops', 0):.4f} GFLOPs")
            print(f"  Total (all {self.model.num_experts} experts): {flops.get('total_gflops', 0):.4f} GFLOPs")
            print(f"  Per sample: {flops.get('gflops_per_sample', 0):.6f} GFLOPs")
            if 'batch_size' in flops:
                print(f"  Batch size used: {flops['batch_size']}")

        # Memory
        if 'memory' in self.metrics and self.metrics['memory']:
            mem = self.metrics['memory']
            if 'error' not in mem:
                print("\n[MEMORY] GPU Memory Usage:")
                print(f"  Model parameters: {mem.get('model_params_mb', 0):.2f} MB")
                print(f"  Forward activations: {mem.get('forward_activations_mb', 0):.2f} MB")
                print(f"  Backward gradients: {mem.get('backward_gradients_mb', 0):.2f} MB")
                print(f"  Peak GPU memory: {mem.get('peak_memory_mb', 0):.2f} MB")
                print(f"  Total allocated: {mem.get('total_allocated_mb', 0):.2f} MB")

        # Timing
        if 'timing' in self.metrics and self.metrics['timing']:
            timing = self.metrics['timing']
            print("\n[TIMING] Training Performance:")
            print(f"  Epoch time: {timing.get('epoch_time', 0):.2f} seconds")
            print(f"  Data loading: {timing.get('data_time', 0):.2f}s ({timing.get('data_fraction', 0)*100:.1f}%)")
            print(f"  Computation: {timing.get('compute_time', 0):.2f}s ({timing.get('compute_fraction', 0)*100:.1f}%)")
            print(f"  Throughput: {timing.get('samples_per_second', 0):.2f} samples/sec")
            print(f"  Avg batch time: {timing.get('avg_batch_time', 0)*1000:.2f} ms")

        print("\n" + "="*80 + "\n")

    def save_summary(self, filepath: Optional[str] = None):
        """Save profiling summary to JSON file"""

        if filepath is None:
            filepath = self.save_dir / 'profiling_summary.json'
        else:
            filepath = Path(filepath)

        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Profiling summary saved to: {filepath}")


def quick_profile(model, input_shape, batch_size=64, device='cuda'):
    """
    Quick profiling for MoE model (all metrics in one call)

    Args:
        model: MoE model to profile
        input_shape: Input tensor shape (without batch dimension)
        batch_size: Batch size for profiling
        device: Device to use

    Returns:
        MoEProfiler instance with all metrics
    """

    profiler = MoEProfiler(model, device)

    print("Profiling model size...")
    profiler.profile_model_size()

    print("Profiling FLOPs...")
    profiler.profile_flops(input_shape, batch_size)

    if torch.cuda.is_available():
        print("Profiling memory usage...")
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        profiler.profile_memory(dummy_input)

    profiler.print_summary()
    profiler.save_summary()

    return profiler


if __name__ == '__main__':
    print("Testing MoEProfiler...")
    print("This requires models.moe_geometric_classifier to be importable.")
    print("Run from the main project directory.\n")
