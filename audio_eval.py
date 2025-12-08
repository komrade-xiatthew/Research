#!/usr/bin/env python3
"""
Audio Evaluation Script - FD, FAD, and KL Metrics

This script computes evaluation metrics for audio quality assessment:
- FD (Frechet Distance): Using PANNs or VGGish features
- FAD (Frechet Audio Distance): Using VGGish features (standard)
- KL (Kullback-Leibler Divergence): Using PANNs features

Based on the AudioEditor paper (ICASSP 2025):
https://arxiv.org/abs/2410.02964

Usage:
    # Compare two audio files
    python audio_eval.py --audio1 generated.wav --audio2 reference.wav

    # Compare two directories
    python audio_eval.py --audio1 outputs/ --audio2 references/

    # Specify device and sample rate (use 'cpu' for Mac)
    python audio_eval.py --audio1 a.wav --audio2 b.wav --device cpu --sr 22050

    # Compute only specific metrics
    python audio_eval.py --audio1 a.wav --audio2 b.wav --metrics fad kl
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from util import (
    calculate_frechet_distance,
    calculate_frechet_audio_distance,
    calculate_kl_divergence
)


def validate_path(path):
    """
    Validate that the path exists and is either a file or directory.

    Args:
        path: Path to validate

    Returns:
        Path object if valid

    Raises:
        ValueError if path doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def print_header():
    """Print a formatted header for the evaluation results."""
    print("\n" + "="*70)
    print("  Audio Evaluation Metrics (AudioEditor Paper)")
    print("="*70)


def print_metric(metric_name, score, description=""):
    """
    Print a metric in a formatted way.

    Args:
        metric_name: Name of the metric (e.g., "FAD")
        score: Numeric score
        description: Optional description
    """
    print(f"\n{metric_name:25s}: {score:.4f}")
    if description:
        print(f"{'':25s}  ({description})")


def compute_all_metrics(audio1, audio2, device='cuda', sr=16000, metrics=None):
    """
    Compute all evaluation metrics between two audio sources.

    Args:
        audio1: Path to first audio file or directory
        audio2: Path to second audio file or directory
        device: 'cuda' or 'cpu'
        sr: Sample rate
        metrics: List of metrics to compute (default: all)

    Returns:
        dict: Dictionary containing all computed metrics
    """
    if metrics is None:
        metrics = ['fd', 'fad', 'kl']

    results = {}

    # Determine input types
    audio1_path = validate_path(audio1)
    audio2_path = validate_path(audio2)

    is_file1 = audio1_path.is_file()
    is_file2 = audio2_path.is_file()

    # Print input information
    print_header()
    print(f"\nInput 1: {audio1}")
    print(f"  Type: {'Audio file' if is_file1 else 'Directory'}")
    if not is_file1:
        wav_files = list(audio1_path.glob('*.wav'))
        print(f"  Files: {len(wav_files)} audio files found")

    print(f"\nInput 2: {audio2}")
    print(f"  Type: {'Audio file' if is_file2 else 'Directory'}")
    if not is_file2:
        wav_files = list(audio2_path.glob('*.wav'))
        print(f"  Files: {len(wav_files)} audio files found")

    print(f"\nDevice: {device}")
    print(f"Sample Rate: {sr} Hz")
    print("\n" + "-"*70)
    print("Computing metrics...")
    print("-"*70)

    # Compute FD (Frechet Distance) using PANNs
    if 'fd' in metrics:
        try:
            print("\n[1/3] Computing FD (Frechet Distance with PANNs)...")
            fd_score = calculate_frechet_distance(
                str(audio1_path),
                str(audio2_path),
                model_name='pann',
                sr=sr,
                device=device
            )
            results['fd'] = fd_score
            print(f"      ✓ FD = {fd_score:.4f}")
        except Exception as e:
            print(f"      ✗ Error computing FD: {e}")
            results['fd'] = None

    # Compute FAD (Frechet Audio Distance) using VGGish
    if 'fad' in metrics:
        try:
            print("\n[2/3] Computing FAD (Frechet Audio Distance with VGGish)...")
            fad_score = calculate_frechet_audio_distance(
                str(audio1_path),
                str(audio2_path),
                sr=sr,
                device=device
            )
            results['fad'] = fad_score
            print(f"      ✓ FAD = {fad_score:.4f}")
        except Exception as e:
            print(f"      ✗ Error computing FAD: {e}")
            results['fad'] = None

    # Compute KL (Kullback-Leibler Divergence) using PANNs
    if 'kl' in metrics:
        try:
            print("\n[3/3] Computing KL (Kullback-Leibler Divergence with PANNs)...")
            kl_score = calculate_kl_divergence(
                str(audio1_path),
                str(audio2_path),
                model_name='pann',
                sr=sr,
                device=device
            )
            results['kl'] = kl_score
            print(f"      ✓ KL = {kl_score:.4f}")
        except Exception as e:
            print(f"      ✗ Error computing KL: {e}")
            results['kl'] = None

    return results


def print_results(results):
    """
    Print final results in a formatted table.

    Args:
        results: Dictionary of metric results
    """
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)

    if results.get('fd') is not None:
        print_metric(
            "FD (Frechet Distance)",
            results['fd'],
            "Lower is better"
        )

    if results.get('fad') is not None:
        print_metric(
            "FAD (Frechet Audio Dist.)",
            results['fad'],
            "Lower is better"
        )

    if results.get('kl') is not None:
        print_metric(
            "KL (KL Divergence)",
            results['kl'],
            "Lower is better, 0 = identical"
        )

    print("\n" + "="*70)
    print("\nInterpretation:")
    print("  - Lower scores indicate more similar audio")
    print("  - FD & FAD measure distribution similarity")
    print("  - KL measures distribution divergence")
    print("="*70 + "\n")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Compute audio evaluation metrics (FD, FAD, KL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two audio files
  python audio_eval.py --audio1 generated.wav --audio2 reference.wav

  # Compare two directories
  python audio_eval.py --audio1 outputs/ --audio2 references/

  # Use CPU instead of GPU (required for Mac)
  python audio_eval.py --audio1 a.wav --audio2 b.wav --device cpu

  # Compute only FAD
  python audio_eval.py --audio1 a.wav --audio2 b.wav --metrics fad

Metrics:
  FD  - Frechet Distance (using PANNs features)
  FAD - Frechet Audio Distance (using VGGish features)
  KL  - Kullback-Leibler Divergence (using PANNs features)

Reference:
  AudioEditor: A Training-Free Diffusion-Based Audio Editing Framework
  ICASSP 2025 - https://arxiv.org/abs/2410.02964
        """
    )

    parser.add_argument(
        '--audio1',
        type=str,
        required=True,
        help='Path to first audio file or directory'
    )

    parser.add_argument(
        '--audio2',
        type=str,
        required=True,
        help='Path to second audio file or directory'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation (default: cuda)'
    )

    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sample rate in Hz (default: 16000)'
    )

    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['fd', 'fad', 'kl'],
        choices=['fd', 'fad', 'kl'],
        help='Metrics to compute (default: all)'
    )

    args = parser.parse_args()

    try:
        # Compute metrics
        results = compute_all_metrics(
            args.audio1,
            args.audio2,
            device=args.device,
            sr=args.sr,
            metrics=args.metrics
        )

        # Print results
        print_results(results)

        # Exit with success
        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
