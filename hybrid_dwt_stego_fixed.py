# hybrid_dwt_stego_fixed.py
# FULLY CORRECTED: Robust DWT coefficient handling with visualization
# NOW SUPPORTS: Binary input/output with hex intermediate conversion
# Requirements: pycryptodome, numpy, pywavelets, opencv-python, scikit-image, matplotlib
import warnings
warnings.filterwarnings("ignore")

import os
import math
import json
import base64
import hashlib
import numpy as np
import pywt
import cv2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Util import number
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Hardcoded optimal embedding strength
EMBEDDING_STRENGTH = 15  # Optimal balance between robustness and imperceptibility

# ========================================
# PART 1: AES ENCRYPTION
# ========================================

def get_key(user_key: str) -> bytes:
    """
    Derive 32-byte AES key from user password
    Supports both TEXT and BINARY password formats
    """
    user_key = user_key.strip()
    
    # Check if password is binary (only 0s and 1s, spaces allowed)
    cleaned_key = user_key.replace(' ', '').replace('_', '')
    
    if all(c in '01' for c in cleaned_key) and len(cleaned_key) >= 8:
        # Binary password detected
        print("  → Detected BINARY password format")
        
        # Pad or truncate to 256 bits (32 bytes)
        if len(cleaned_key) < 256:
            cleaned_key = cleaned_key.ljust(256, '0')
        else:
            cleaned_key = cleaned_key[:256]
        
        # Convert binary string to bytes
        key = int(cleaned_key, 2).to_bytes(32, 'big')
        return key
    else:
        # Text password (original behavior)
        print("  → Detected TEXT password format")
        return user_key.encode('utf-8').ljust(32, b'\0')[:32]

def aes_encrypt_bytes(data_bytes: bytes, key: bytes) -> bytes:
    """Encrypt data using AES-CBC"""
    iv = os.urandom(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(data_bytes, AES.block_size))
    return iv + ct

def aes_decrypt_bytes(enc_bytes: bytes, key: bytes) -> bytes:
    """Decrypt AES-CBC encrypted data"""
    iv = enc_bytes[:AES.block_size]
    ct = enc_bytes[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)

# ========================================
# PART 1B: BINARY/HEX CONVERSION
# ========================================

def binary_to_hex(binary_str: str) -> str:
    """Convert binary string to hex string"""
    binary_str = binary_str.replace(' ', '').replace('_', '')
    if not all(c in '01' for c in binary_str):
        raise ValueError("Input must contain only 0s and 1s")
    padding = (4 - len(binary_str) % 4) % 4
    binary_str = '0' * padding + binary_str
    hex_str = hex(int(binary_str, 2))[2:]
    return hex_str

def hex_to_binary(hex_str: str) -> str:
    """Convert hex string to binary string"""
    hex_str = hex_str.replace('0x', '').replace(' ', '').replace('_', '')
    binary_str = bin(int(hex_str, 16))[2:]
    return binary_str

def binary_to_bytes(binary_str: str) -> bytes:
    """Convert binary string to bytes via hex"""
    hex_str = binary_to_hex(binary_str)
    if len(hex_str) % 2 != 0:
        hex_str = '0' + hex_str
    return bytes.fromhex(hex_str)

def bytes_to_binary(data_bytes: bytes) -> str:
    """Convert bytes to binary string via hex"""
    hex_str = data_bytes.hex()
    return hex_to_binary(hex_str)

# ========================================
# PART 2: ARNOLD CHAOTIC MAP
# ========================================

def arnold_map_once(arr: np.ndarray) -> np.ndarray:
    N = arr.shape[0]
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    XP = (X + Y) % N
    YP = (X + 2*Y) % N
    out = np.empty_like(arr)
    out[YP, XP] = arr[Y, X]
    return out

def arnold_map_inverse_once(arr: np.ndarray) -> np.ndarray:
    N = arr.shape[0]
    Xp, Yp = np.meshgrid(np.arange(N), np.arange(N))
    X = (2*Xp - Yp) % N
    Y = (-Xp + Yp) % N
    out = np.empty_like(arr)
    out[Y, X] = arr[Yp, Xp]
    return out

def scramble_bytes(enc_bytes: bytes, iterations: int) -> str:
    L = len(enc_bytes)
    N = math.ceil(math.sqrt(L))
    pad_len = N*N - L
    padded = enc_bytes + (os.urandom(pad_len) if pad_len else b'')
    arr = np.frombuffer(padded, dtype=np.uint8).reshape((N, N))
    temp = arr.copy()
    for _ in range(iterations):
        temp = arnold_map_once(temp)
    scrambled = temp.tobytes()
    header = {"N": N, "iterations": iterations, "pad_len": pad_len}
    pkg = json.dumps(header) + "::" + base64.b64encode(scrambled).decode('ascii')
    return pkg

def unscramble_bytes(pkg: str) -> bytes:
    header_json, payload = pkg.split("::", 1)
    h = json.loads(header_json)
    N, iterations, pad_len = h["N"], h["iterations"], h["pad_len"]
    scrambled = base64.b64decode(payload)
    arr = np.frombuffer(scrambled, dtype=np.uint8).reshape((N, N))
    temp = arr.copy()
    for _ in range(iterations):
        temp = arnold_map_inverse_once(temp)
    data_padded = temp.tobytes()
    data = data_padded[:-pad_len] if pad_len else data_padded
    return data

# ========================================
# PART 3: HYBRID BBS + HASH GENERATOR
# ========================================

class HybridPositionGenerator:
    def __init__(self, key: bytes, block_id: tuple):
        self.key = key
        self.block_id = block_id
        seed_material = key + str(block_id).encode()
        self.seed_hash = hashlib.sha256(seed_material).digest()
    
    def generate_positions(self, band_size: int, num_positions: int) -> list:
        positions = []
        seen = set()
        counter = 0
        while len(positions) < num_positions and counter < num_positions * 10:
            hash_input = self.seed_hash + counter.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            index = int.from_bytes(hash_output[:4], 'big') % band_size
            if index not in seen:
                positions.append(index)
                seen.add(index)
            counter += 1
        if len(positions) < num_positions:
            all_pos = set(range(band_size))
            available = sorted(list(all_pos - seen))
            positions.extend(available[:num_positions - len(positions)])
        return positions[:num_positions]

# ========================================
# PART 4: BIT MANIPULATION
# ========================================

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def bits_to_bytes(bits: list) -> bytes:
    while len(bits) % 8 != 0:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        val = 0
        for bit in chunk:
            val = (val << 1) | (bit & 1)
        out.append(val)
    return bytes(out)

def int_to_32bits(n: int) -> list:
    return [(n >> (31 - i)) & 1 for i in range(32)]

def bits32_to_int(bits32: list) -> int:
    n = 0
    for b in bits32[:32]:
        n = (n << 1) | (b & 1)
    return n

# ========================================
# PART 5: ENERGY-BASED BLOCK SELECTION
# ========================================

def calculate_block_energy(block: np.ndarray) -> float:
    return np.var(block.astype(np.float32))

def select_high_energy_blocks(channel: np.ndarray, block_size: int, 
                              threshold_percentile: float = 50.0) -> list:
    h, w = channel.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    energies = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = channel[i*block_size:(i+1)*block_size, 
                          j*block_size:(j+1)*block_size]
            energy = calculate_block_energy(block)
            energies.append(((i, j), energy))
    threshold = np.percentile([e[1] for e in energies], threshold_percentile)
    selected = sorted([pos for pos, energy in energies if energy >= threshold])
    return selected

# ========================================
# PART 6: DWT EMBEDDING
# ========================================

def dwt2_block(block: np.ndarray, wavelet='haar') -> tuple:
    coeffs = pywt.dwt2(block.astype(np.float32), wavelet, mode='periodization')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH

def idwt2_block(LL, LH, HL, HH, wavelet='haar') -> np.ndarray:
    rec = pywt.idwt2((LL, (LH, HL, HH)), wavelet, mode='periodization')
    return np.clip(np.rint(rec), 0, 255).astype(np.uint8)

def embed_bits_robust(band: np.ndarray, positions: list, bits: list, strength: int = EMBEDDING_STRENGTH) -> tuple:
    band_modified = band.copy()
    band_flat = band_modified.flatten()
    original_values = {}
    for pos, bit in zip(positions, bits):
        if pos >= len(band_flat):
            break
        coef = float(band_flat[pos])
        original_values[pos] = float(coef)
        if bit == 1:
            new_coef = coef + strength
        else:
            new_coef = coef - strength
        band_flat[pos] = new_coef
    return band_flat.reshape(band.shape), original_values

def extract_bits_robust(band: np.ndarray, positions: list, original_values: dict, strength: int = EMBEDDING_STRENGTH) -> list:
    band_flat = band.flatten()
    bits = []
    for pos in positions:
        if pos < len(band_flat):
            coef = float(band_flat[pos])
            if pos in original_values:
                original = original_values[pos]
                delta = coef - original
                if delta > 0:
                    bit = 1
                else:
                    bit = 0
            else:
                bit = 1 if abs(coef) > strength else 0
            bits.append(bit)
        else:
            bits.append(0)
    return bits

# ========================================
# PART 7: MULTI-LAYER EMBEDDER
# ========================================

class MultiLayerEmbedder:
    def __init__(self, key: bytes, block_size: int = 16, wavelet: str = 'haar', strength: int = EMBEDDING_STRENGTH):
        self.key = key
        self.block_size = block_size
        self.wavelet = wavelet
        self.strength = strength
        self.bands_to_use = ['HL', 'LH', 'HH']
        self.channel_priority = {'B': 0.70, 'G': 0.25, 'R': 0.05}
    
    def embed(self, img_bgr: np.ndarray, payload_bits: list) -> tuple:
        h, w = img_bgr.shape[:2]
        stego = img_bgr.copy()
        total_bits = len(payload_bits)
        blue_bits = int(total_bits * self.channel_priority['B'])
        green_bits = int(total_bits * self.channel_priority['G'])
        red_bits = total_bits - blue_bits - green_bits
        bit_allocation = {
            2: payload_bits[:blue_bits],
            1: payload_bits[blue_bits:blue_bits+green_bits],
            0: payload_bits[blue_bits+green_bits:]
        }
        embedding_map = {}
        for channel_idx, channel_name in [(2, 'B'), (1, 'G'), (0, 'R')]:
            bits = bit_allocation[channel_idx]
            if not bits:
                continue
            channel = img_bgr[:, :, channel_idx].astype(np.float32)
            high_energy_blocks = select_high_energy_blocks(channel, self.block_size, threshold_percentile=60)
            bits_embedded = 0
            embedding_map[channel_name] = []
            for block_row, block_col in high_energy_blocks:
                if bits_embedded >= len(bits):
                    break
                r_start = block_row * self.block_size
                r_end = r_start + self.block_size
                c_start = block_col * self.block_size
                c_end = c_start + self.block_size
                block = channel[r_start:r_end, c_start:c_end]
                LL, LH, HL, HH = dwt2_block(block, self.wavelet)
                bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
                for band_name in self.bands_to_use:
                    if bits_embedded >= len(bits):
                        break
                    band = bands[band_name]
                    band_capacity = band.size
                    block_id = (block_row, block_col, channel_name, band_name)
                    pos_gen = HybridPositionGenerator(self.key, block_id)
                    bits_to_embed = min(len(bits) - bits_embedded, band_capacity // 4)
                    if bits_to_embed > 0:
                        positions = pos_gen.generate_positions(band_capacity, bits_to_embed)
                        bits_chunk = bits[bits_embedded:bits_embedded + bits_to_embed]
                        bands[band_name], original_vals = embed_bits_robust(band, positions, bits_chunk, self.strength)
                        embedding_map[channel_name].append({
                            'block': (block_row, block_col),
                            'band': band_name,
                            'num_bits': bits_to_embed,
                            'original_values': original_vals
                        })
                        bits_embedded += bits_to_embed
                reconstructed = idwt2_block(bands['LL'], bands['LH'], bands['HL'], bands['HH'], self.wavelet)
                stego[r_start:r_end, c_start:c_end, channel_idx] = reconstructed
        return stego, embedding_map
    
    def extract(self, stego_bgr: np.ndarray, embedding_map: dict, total_bits: int) -> list:
        extracted_bits = []
        for channel_idx, channel_name in [(2, 'B'), (1, 'G'), (0, 'R')]:
            if channel_name not in embedding_map:
                continue
            channel = stego_bgr[:, :, channel_idx].astype(np.float32)
            for entry in embedding_map[channel_name]:
                block_row, block_col = entry['block']
                band_name = entry['band']
                num_bits = entry['num_bits']
                original_values = entry.get('original_values', {})
                original_values = {int(k): v for k, v in original_values.items()}
                r_start = block_row * self.block_size
                r_end = r_start + self.block_size
                c_start = block_col * self.block_size
                c_end = c_start + self.block_size
                block = channel[r_start:r_end, c_start:c_end]
                LL, LH, HL, HH = dwt2_block(block, self.wavelet)
                bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
                block_id = (block_row, block_col, channel_name, band_name)
                pos_gen = HybridPositionGenerator(self.key, block_id)
                positions = pos_gen.generate_positions(bands[band_name].size, num_bits)
                bits = extract_bits_robust(bands[band_name], positions, original_values, self.strength)
                extracted_bits.extend(bits)
        return extracted_bits[:total_bits]

# ========================================
# PART 8: VISUALIZATION
# ========================================

def calculate_ssim(cover_path: str, stego_path: str) -> float:
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    if cover is None or stego is None:
        return 0.0
    cover_gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(cover_gray, stego_gray, data_range=255)
    return ssim_value

def visualize_comparison(cover_path: str, stego_path: str):
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)
    if cover is None or stego is None:
        print("⚠ Cannot load images for visualization")
        return
    cover_rgb = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
    ssim_value = calculate_ssim(cover_path, stego_path)
    diff = cv2.absdiff(cover, stego)
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    pixels_changed = np.count_nonzero(diff)
    total_pixels = diff.shape[0] * diff.shape[1]
    percent_changed = (pixels_changed / total_pixels) * 100
    if max_diff > 0:
        amplification = min(50, 255 / max_diff)
    else:
        amplification = 50
    diff_amplified = np.clip(diff_rgb * amplification, 0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cover_rgb)
    ax1.set_title('Cover Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(stego_rgb)
    ax2.set_title('Stego Image', fontsize=14, fontweight='bold')
    ax2.axis('off')
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(diff_amplified)
    ax3.set_title(f'Difference ({amplification:.1f}x amplified)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Pixel difference')
    ax4 = fig.add_subplot(gs[1, :])
    diff_flat = diff.flatten()
    diff_flat_nonzero = diff_flat[diff_flat > 0]
    if len(diff_flat_nonzero) > 0:
        ax4.hist(diff_flat_nonzero, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Pixel Difference Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Non-Zero Pixel Differences', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.2f}')
        ax4.axvline(max_diff, color='orange', linestyle='--', linewidth=2, label=f'Max: {max_diff:.0f}')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No pixel differences detected!', ha='center', va='center', fontsize=16, transform=ax4.transAxes)
        ax4.axis('off')
    stats_text = f'SSIM: {ssim_value:.6f} | Max Diff: {max_diff:.0f} | Mean Diff: {mean_diff:.2f} | Changed Pixels: {percent_changed:.2f}%'
    fig.suptitle(stats_text, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    print("\n" + "="*60)
    print("IMAGE QUALITY METRICS")
    print("="*60)
    print(f"SSIM: {ssim_value:.6f} (1.0=perfect, >0.95=excellent)")
    mse = np.mean((cover.astype(float) - stego.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255 ** 2 / mse)
    else:
        psnr = float('inf')
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB (>40=excellent, >30=good)")
    print("="*60 + "\n")

# ========================================
# PART 9: MAIN PIPELINE (BINARY I/O)
# ========================================

def embed_message(cover_path: str, stego_path: str, binary_message: str, 
                 password: str, arnold_iterations: int = 5, block_size: int = 16, wavelet: str = 'haar'):
    print("\n" + "="*60)
    print("EMBEDDING PIPELINE (Binary Input)")
    print("="*60)
    img = cv2.imread(cover_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {cover_path}")
    print(f"✓ Loaded cover image: {img.shape}")
    print(f"✓ Binary message length: {len(binary_message)} bits")
    message_bytes = binary_to_bytes(binary_message)
    print(f"✓ Converted to: {len(message_bytes)} bytes (via hex)")
    print(f"✓ Processing password...")
    key = get_key(password)
    encrypted = aes_encrypt_bytes(message_bytes, key)
    print(f"✓ AES encrypted: {len(encrypted)} bytes")
    scrambled_pkg = scramble_bytes(encrypted, arnold_iterations)
    payload_bytes = scrambled_pkg.encode('utf-8')
    print(f"✓ Arnold scrambled ({arnold_iterations} iter): {len(payload_bytes)} bytes")
    payload_len = len(payload_bytes)
    header_bits = int_to_32bits(payload_len)
    payload_bits = bytes_to_bits(payload_bytes)
    bitstream = header_bits + payload_bits
    print(f"✓ Bitstream: {len(bitstream)} bits (32 header + {len(payload_bits)} payload)")
    embedder = MultiLayerEmbedder(key, block_size, wavelet, EMBEDDING_STRENGTH)
    stego, embed_map = embedder.embed(img, bitstream)
    print(f"✓ Embedded using {block_size}x{block_size} blocks, strength={EMBEDDING_STRENGTH}")
    cv2.imwrite(stego_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    map_path = stego_path.replace('.png', '_map.json')
    with open(map_path, 'w') as f:
        json.dump({
            'embedding_map': embed_map,
            'total_bits': len(bitstream),
            'payload_length': payload_len,
            'block_size': block_size,
            'wavelet': wavelet,
            'arnold_iterations': arnold_iterations,
            'strength': EMBEDDING_STRENGTH,
            'cover_path': cover_path,
            'original_binary_length': len(binary_message)
        }, f, indent=2)
    print(f"✓ Stego saved: {stego_path}")
    print(f"✓ Map saved: {map_path}")
    print("="*60 + "\n")
    visualize_comparison(cover_path, stego_path)
    return stego, embed_map

def extract_message(stego_path: str, password: str) -> str:
    print("\n" + "="*60)
    print("EXTRACTION PIPELINE (Binary Output)")
    print("="*60)
    stego = cv2.imread(stego_path)
    if stego is None:
        raise FileNotFoundError(f"Cannot load image: {stego_path}")
    map_path = stego_path.replace('.png', '_map.json')
    with open(map_path, 'r') as f:
        data = json.load(f)
    embedding_map = data['embedding_map']
    total_bits = data['total_bits']
    payload_length = data['payload_length']
    block_size = data['block_size']
    wavelet = data['wavelet']
    arnold_iterations = data['arnold_iterations']
    strength = data.get('strength', EMBEDDING_STRENGTH)
    cover_path = data.get('cover_path', None)
    original_binary_length = data.get('original_binary_length', None)
    print(f"✓ Loaded stego and map")
    print(f"✓ Expected: {payload_length} bytes, {total_bits} bits")
    print(f"✓ Processing password...")
    key = get_key(password)
    embedder = MultiLayerEmbedder(key, block_size, wavelet, strength)
    bitstream = embedder.extract(stego, embedding_map, total_bits)
    print(f"✓ Extracted {len(bitstream)} bits")
    header_bits = bitstream[:32]
    payload_bits = bitstream[32:]
    extracted_len = bits32_to_int(header_bits)
    print(f"✓ Header: {extracted_len} bytes")
    if abs(extracted_len - payload_length) > 10:
        print(f"⚠ Mismatch! Using map: {payload_length}")
        extracted_len = payload_length
    bits_needed = extracted_len * 8
    payload_bits = payload_bits[:bits_needed]
    payload_bytes = bits_to_bytes(payload_bits)
    print(f"✓ Converted to {len(payload_bytes)} bytes")
    try:
        scrambled_pkg = payload_bytes.decode('utf-8')
        encrypted = unscramble_bytes(scrambled_pkg)
        print(f"✓ Arnold unscrambled ({arnold_iterations} iter)")
        message_bytes = aes_decrypt_bytes(encrypted, key)
        print(f"✓ AES decrypted")
        binary_message = bytes_to_binary(message_bytes)
        if original_binary_length:
            binary_message = binary_message[-original_binary_length:]
        print(f"✓ Converted to binary: {len(binary_message)} bits")
        print("="*60 + "\n")
        if cover_path and os.path.exists(cover_path):
            visualize_comparison(cover_path, stego_path)
        return binary_message
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Hex: {payload_bytes[:50].hex()}")
        raise

# ========================================
# PART 10: INTERACTIVE INTERFACE
# ========================================

def main():
    print("\n" + "="*60)
    print("HYBRID DWT STEGANOGRAPHY - BINARY I/O")
    print("Password Support: TEXT or BINARY (auto-detected)")
    print("="*60)
    mode = input("\nMode (1=Embed, 2=Extract): ").strip()
    if mode == '1':
        cover_path = input("Cover image path: ").strip().strip('"').strip("'")
        stego_path = input("Output stego path (e.g., stego.png): ").strip()
        binary_message = input("Binary message (0s and 1s): ").strip()
        print("\nPassword Format Options:")
        print("  • TEXT: e.g., 'MySecurePass123'")
        print("  • BINARY: e.g., '10110101...' (256 bits recommended)")
        password = input("Password: ")
        arnold_iter = input("Arnold iterations (default=5): ").strip()
        arnold_iter = int(arnold_iter) if arnold_iter else 5
        block_size = input("Block size 8/16/32 (default=16): ").strip()
        block_size = int(block_size) if block_size else 16
        try:
            stego, _ = embed_message(cover_path, stego_path, binary_message, password, arnold_iter, block_size)
            print("✓ SUCCESS: Message embedded!")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    elif mode == '2':
        stego_path = input("Stego image path: ").strip().strip('"').strip("'")
        print("\nPassword Format Options:")
        print("  • TEXT: e.g., 'MySecurePass123'")
        print("  • BINARY: e.g., '10110101...' (same as embedding)")
        password = input("Password: ")
        try:
            binary_message = extract_message(stego_path, password)
            print(f"\n{'='*60}")
            print("EXTRACTED BINARY MESSAGE:")
            print(f"{'='*60}")
            print(binary_message)
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()
