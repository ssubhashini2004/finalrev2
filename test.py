"""
Steganography Testing Script - Automated SNR/PSNR/SSIM Analysis
Tests multiple message lengths and generates plots
IMPROVED: Better error handling + SSIM plot
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# Import your steganography module
# Assuming hybrid_dwt_stego_fixed.py is in the same directory
try:
    from hybrid_dwt_stego_fixed import embed_message, extract_message, get_key
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    print(f"⚠ Import error: {e}")
    print("Place this script in the same folder as hybrid_dwt_stego_fixed.py")
    exit(1)


def calculate_metrics(cover_path, stego_path):
    """Calculate SNR, PSNR, MSE, and SSIM"""
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    
    if cover is None or stego is None:
        return None, None, None, None
    
    # Convert to float for accurate calculation
    cover_f = cover.astype(np.float64)
    stego_f = stego.astype(np.float64)
    
    # MSE (Mean Squared Error)
    mse = np.mean((cover_f - stego_f) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)
    
    # SNR (Signal-to-Noise Ratio)
    signal_power = np.mean(cover_f ** 2)
    noise_power = mse
    if noise_power == 0:
        snr = float('inf')
    else:
        snr = 10 * np.log10(signal_power / noise_power)
    
    # SSIM (Structural Similarity Index)
    cover_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(cover_gray, stego_gray, data_range=255)
    
    return snr, psnr, mse, ssim_value


def estimate_capacity(image_path, block_size=16):
    """Estimate maximum capacity for an image"""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    h, w = img.shape[:2]
    blocks = (h // block_size) * (w // block_size)
    high_energy_blocks = int(blocks * 0.4)  # 40% selection
    
    # Conservative estimate: 64 bits per block (3 bands × ~21 positions)
    estimated_bits = high_energy_blocks * 64 * 0.7  # 70% safety margin
    
    # Account for overhead
    overhead_bytes = 100  # AES IV + Arnold header + padding
    usable_bits = estimated_bits - (overhead_bytes * 8)
    
    return max(0, int(usable_bits))


def generate_test_messages():
    """Generate binary test messages of increasing lengths"""
    # Test lengths from small to large (in bits)
    test_lengths = [
        100,          # 10^2 - Very small
        500,          # Small
        1000,         # 10^3 - 1 KB
        2000,         # 2 KB
        5000,         # 5 KB
        10000,        # 10^4 - 10 KB
        20000,        # 20 KB
        50000,        # 50 KB
        100000,       # 10^5 - 100 KB
        200000,       # 200 KB
        500000,       # 10^5.7 - 500 KB
        1000000,      # 10^6 - 1 MB
        2000000,      # 2 MB
        5000000,      # 5 MB
        10000000,     # 10^7 - 10 MB (if capacity allows)
    ]
    
    messages = {}
    for length in test_lengths:
        # Generate alternating binary pattern for consistency
        binary_msg = ''.join(['1' if i % 2 == 0 else '0' for i in range(length)])
        messages[length] = binary_msg
    
    return messages


def run_tests(cover_image_path, password="TestPassword123", output_dir="test_results"):
    """Run comprehensive tests and collect metrics"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Estimate capacity
    max_capacity = estimate_capacity(cover_image_path)
    print(f"\n{'='*70}")
    print(f"ESTIMATED IMAGE CAPACITY: {max_capacity} bits ({max_capacity//8} bytes)")
    print(f"{'='*70}\n")
    
    # Generate test messages
    all_messages = generate_test_messages()
    
    # Filter messages that fit in capacity
    test_messages = {k: v for k, v in all_messages.items() if k <= max_capacity}
    
    if not test_messages:
        print("⚠ No test messages fit in image capacity!")
        print(f"   Try a larger image. Current capacity: {max_capacity} bits")
        return None
    
    print(f"Testing {len(test_messages)} message lengths (up to {max(test_messages.keys())} bits)\n")
    
    results = {
        'lengths': [],
        'snr': [],
        'psnr': [],
        'mse': [],
        'ssim': [],
        'embed_time': [],
        'extract_time': [],
        'success': []
    }
    
    for length, binary_msg in sorted(test_messages.items()):
        print(f"\n{'='*70}")
        print(f"Testing: {length} bits ({length/8:.0f} bytes, 10^{np.log10(length):.2f})")
        print(f"{'='*70}")
        
        stego_path = os.path.join(output_dir, f"stego_{length}bits.png")
        
        try:
            # Check if message fits (safety check before embedding)
            if length > max_capacity:
                print(f"⚠ Cover image not big enough to embed message of size {length} bits")
                print(f"   Maximum capacity: {max_capacity} bits ({max_capacity//8} bytes)")
                print(f"   Required: {length} bits ({length//8} bytes)")
                print(f"   Shortfall: {length - max_capacity} bits")
                break
            
            # EMBEDDING
            start_time = time.time()
            embed_message(
                cover_path=cover_image_path,
                stego_path=stego_path,
                binary_message=binary_msg,
                password=password,
                arnold_iterations=3,  # Reduce for speed
                block_size=16
            )
            embed_time = time.time() - start_time
            print(f"✓ Embedding time: {embed_time:.2f}s")
            
            # EXTRACTION
            start_time = time.time()
            extracted = extract_message(stego_path, password)
            extract_time = time.time() - start_time
            print(f"✓ Extraction time: {extract_time:.2f}s")
            
            # VERIFICATION
            success = (extracted == binary_msg)
            if success:
                print(f"✓ Message integrity: PERFECT MATCH")
            else:
                print(f"✗ Message integrity: MISMATCH (first 100 bits)")
                print(f"   Original:  {binary_msg[:100]}")
                print(f"   Extracted: {extracted[:100]}")
            
            # CALCULATE METRICS
            snr_val, psnr_val, mse_val, ssim_val = calculate_metrics(cover_image_path, stego_path)
            
            print(f"\nMetrics:")
            print(f"  SNR:  {snr_val:.2f} dB")
            print(f"  PSNR: {psnr_val:.2f} dB")
            print(f"  MSE:  {mse_val:.4f}")
            print(f"  SSIM: {ssim_val:.6f}")
            
            # Store results
            results['lengths'].append(length)
            results['snr'].append(snr_val)
            results['psnr'].append(psnr_val)
            results['mse'].append(mse_val)
            results['ssim'].append(ssim_val)
            results['embed_time'].append(embed_time)
            results['extract_time'].append(extract_time)
            results['success'].append(success)
            
        except ValueError as e:
            # Catch capacity exceeded errors gracefully
            if "capacity" in str(e).lower() or "insufficient" in str(e).lower():
                print(f"⚠ Cover image not big enough to embed message of size {length} bits")
                print(f"   Maximum capacity: {max_capacity} bits ({max_capacity//8} bytes)")
                print(f"   Error details: {str(e)}")
                break
            else:
                print(f"✗ FAILED: {str(e)}")
                raise
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            print(f"   This may indicate capacity exceeded at {length} bits")
            import traceback
            traceback.print_exc()
            break
    
    return results


def plot_results(results, output_dir="test_results"):
    """Generate comprehensive plots including SSIM"""
    
    if not results or len(results['lengths']) == 0:
        print("No results to plot!")
        return
    
    # Create 2x3 grid for 6 plots (was 2x2, now 2x3)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Steganography Performance Analysis', fontsize=18, fontweight='bold')
    
    lengths = results['lengths']
    log_lengths = [np.log10(l) for l in lengths]
    
    # Plot 1: SNR vs Message Length
    ax1 = axes[0, 0]
    ax1.plot(log_lengths, results['snr'], 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.set_xlabel('Message Length (log10 bits)', fontsize=12)
    ax1.set_ylabel('SNR (dB)', fontsize=12)
    ax1.set_title('Signal-to-Noise Ratio vs Message Length', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=40, color='green', linestyle='--', label='Excellent (>40 dB)')
    ax1.axhline(y=30, color='orange', linestyle='--', label='Good (>30 dB)')
    ax1.legend()
    
    # Plot 2: PSNR vs Message Length
    ax2 = axes[0, 1]
    ax2.plot(log_lengths, results['psnr'], 'o-', color='coral', linewidth=2, markersize=8)
    ax2.set_xlabel('Message Length (log10 bits)', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Peak Signal-to-Noise Ratio vs Message Length', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=40, color='green', linestyle='--', label='Excellent (>40 dB)')
    ax2.axhline(y=30, color='orange', linestyle='--', label='Acceptable (>30 dB)')
    ax2.legend()
    
    # Plot 3: SSIM vs Message Length (NEW!)
    ax3 = axes[0, 2]
    ax3.plot(log_lengths, results['ssim'], 'o-', color='darkgreen', linewidth=2, markersize=8)
    ax3.set_xlabel('Message Length (log10 bits)', fontsize=12)
    ax3.set_ylabel('SSIM', fontsize=12)
    ax3.set_title('Structural Similarity Index vs Message Length', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.95, color='green', linestyle='--', label='Excellent (>0.95)')
    ax3.axhline(y=0.90, color='orange', linestyle='--', label='Good (>0.90)')
    ax3.axhline(y=0.80, color='red', linestyle='--', label='Acceptable (>0.80)')
    ax3.set_ylim([0.7, 1.01])  # Focus on relevant range
    ax3.legend()
    
    # Plot 4: MSE vs Message Length
    ax4 = axes[1, 0]
    ax4.plot(log_lengths, results['mse'], 'o-', color='crimson', linewidth=2, markersize=8)
    ax4.set_xlabel('Message Length (log10 bits)', fontsize=12)
    ax4.set_ylabel('Mean Squared Error', fontsize=12)
    ax4.set_title('MSE vs Message Length (Lower is Better)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Processing Time
    ax5 = axes[1, 1]
    ax5.plot(log_lengths, results['embed_time'], 'o-', color='purple', 
             linewidth=2, markersize=8, label='Embedding Time')
    ax5.plot(log_lengths, results['extract_time'], 's-', color='teal', 
             linewidth=2, markersize=8, label='Extraction Time')
    ax5.set_xlabel('Message Length (log10 bits)', fontsize=12)
    ax5.set_ylabel('Time (seconds)', fontsize=12)
    ax5.set_title('Processing Time vs Message Length', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Success Rate / Summary
    ax6 = axes[1, 2]
    success_count = sum(results['success'])
    total_tests = len(results['success'])
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    # Create text summary
    ax6.axis('off')
    summary_text = f"""
    TEST SUMMARY
    ═══════════════════════════
    
    Total Tests: {total_tests}
    Successful: {success_count}
    Failed: {total_tests - success_count}
    Success Rate: {success_rate:.1f}%
    
    AVERAGE METRICS
    ───────────────────────────
    Avg SNR:  {np.mean(results['snr']):.2f} dB
    Avg PSNR: {np.mean(results['psnr']):.2f} dB
    Avg SSIM: {np.mean(results['ssim']):.4f}
    Avg MSE:  {np.mean(results['mse']):.4f}
    
    RANGE TESTED
    ───────────────────────────
    Min: {min(lengths):,} bits
    Max: {max(lengths):,} bits
    Span: 10^{np.log10(min(lengths)):.1f} to 10^{np.log10(max(lengths)):.1f}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.show()
    
    # Save numerical results (updated with SSIM)
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("STEGANOGRAPHY PERFORMANCE RESULTS\n")
        f.write("="*90 + "\n\n")
        f.write(f"{'Length (bits)':<15} {'SNR (dB)':<12} {'PSNR (dB)':<12} {'SSIM':<12} {'MSE':<12} {'Success'}\n")
        f.write("-"*90 + "\n")
        for i in range(len(results['lengths'])):
            f.write(f"{results['lengths'][i]:<15} "
                   f"{results['snr'][i]:<12.2f} "
                   f"{results['psnr'][i]:<12.2f} "
                   f"{results['ssim'][i]:<12.6f} "
                   f"{results['mse'][i]:<12.4f} "
                   f"{'✓' if results['success'][i] else '✗'}\n")
        
        # Add summary statistics
        f.write("\n" + "="*90 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*90 + "\n")
        f.write(f"Total Tests: {len(results['lengths'])}\n")
        f.write(f"Successful: {sum(results['success'])}\n")
        f.write(f"Failed: {len(results['success']) - sum(results['success'])}\n")
        f.write(f"Success Rate: {sum(results['success'])/len(results['success'])*100:.1f}%\n\n")
        f.write(f"Average SNR:  {np.mean(results['snr']):.2f} dB\n")
        f.write(f"Average PSNR: {np.mean(results['psnr']):.2f} dB\n")
        f.write(f"Average SSIM: {np.mean(results['ssim']):.6f}\n")
        f.write(f"Average MSE:  {np.mean(results['mse']):.4f}\n")
    
    print(f"✓ Results saved: {results_path}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("AUTOMATED STEGANOGRAPHY TESTING SUITE")
    print("Enhanced with SSIM metrics and better error handling")
    print("="*70)
    
    # Get cover image
    cover_image = input("\nEnter cover image path (or press Enter for 'cover.png'): ").strip()
    if not cover_image:
        cover_image = "cover.png"
    
    cover_image = cover_image.strip('"').strip("'")
    
    if not os.path.exists(cover_image):
        print(f"\n✗ Error: Image not found: {cover_image}")
        print("\nRECOMMENDED IMAGE SIZES:")
        print("  • For 10^5 bits (100 KB):  1920×1080 (Full HD)")
        print("  • For 10^6 bits (1 MB):    3840×2160 (4K)")
        print("  • For 10^7 bits (10 MB):   12000×8000 (96 MP) - VERY LARGE!")
        return
    
    # Get password
    password = input("Enter password (or press Enter for 'TestPassword123'): ").strip()
    if not password:
        password = "TestPassword123"
    
    # Run tests
    print("\nStarting comprehensive tests...")
    results = run_tests(cover_image, password)
    
    if results:
        plot_results(results)
        print("\n" + "="*70)
        print("TESTING COMPLETE!")
        print("="*70)
        print(f"✓ Tested {len(results['lengths'])} message lengths")
        print(f"✓ Range: {min(results['lengths'])} to {max(results['lengths'])} bits")
        print(f"✓ Results saved in: test_results/")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("NO TESTS COMPLETED")
        print("="*70)
        print("The cover image is too small for any test messages.")
        print("Please use a larger image.")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
