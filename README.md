<div align="center">

# 🌙✨ Hybrid DWT Steganography  
### *Secure • Chaotic • Wavelet-Driven • Deterministic Position Generation*
A modern steganography system blending **DWT**, **AES-CBC**, **Arnold chaos**, and **SHA-256 deterministic embedding** — designed for high security *and* high imperceptibility - embedded across R,G and B channels : 75%,20% and 5% respectively.  

<br>

<img src="/cover.png" width="350">  
<sub><em>Cover image — your secret deserves better than LSB 😉</em></sub>

</div>

---

## 🌸 1. Overview
This project implements a **hybrid, cryptography-enhanced steganography** method with:

✨ Wavelet-domain embedding (HL, LH, HH bands)  
✨ AES-CBC encryption with password support  
✨ Arnold scrambling for chaotic diffusion  
✨ SHA-256 for deterministic bit placement  
✨ Block energy analysis for imperceptible embedding  

Everything is designed for **secure, robust, high-quality steganography**.

---

## 🌿 2. Features 
- 🎨 *Multi-band DWT embedding*  
- 🔐 *AES-CBC payload encryption*  
- 🌀 *Arnold spatiotemporal scrambling*  
- 🌐 *SHA-256–driven embedding map*  
- 🌈 *Energy-aware block selection for minimal distortion*  
- 📊 *Evaluation metrics — PSNR, SSIM, SNR, MSE*  
- 🧪 *Automated testing suite & visualization*

---

## 🧠 3. System Flow  

```
  Message → AES Encryption → Arnold Scramble → Bitstream Packing
                  ↓
         Energy-Based Block Selection
                  ↓
          DWT Transform (HL/LH/HH)
                  ↓
     SHA-256 Position Generator (Deterministic)
                  ↓
         Embedding → Stego Image (PNG)
```

---


---

## 🌼 4. Usage

### ▶ Embedding
```python
from hybrid_dwt_stego_fixed import embed_message

embed_message(
    cover_path="cover.png",
    stego_path="stego.png",
    binary_message="010101...",
    password="MySecretPass",
    arnold_iterations=5,
    block_size=16
)
```

### ▶ Extraction
```python
from hybrid_dwt_stego_fixed import extract_message
msg = extract_message("stego.png", password="MySecretPass")
print(msg)
```

---

## 🌙 5. Visual Comparison

<p align="center">
  <img src="/cover.png" width="380">
  <img src="/stego.png" width="380">
</p>

<p align="center"><em>Left: Cover | Right: Stego (spot the difference... you won't 👀)</em></p>

---

## 🌈 7. Performance Metrics  
> *Security + Quality + Speed = chef’s kiss 💅*

| Size (bits) | SNR (dB) | PSNR (dB) | SSIM | MSE | Emb. Time (s) | Ext. Time (s) | Status |
|-------------|----------|-----------|-------|---------|----------------|----------------|---------|
| 100 | 61.92 | 74.80 | 0.999990 | 0.0022 | 491.39 | 39.45 | Success |
| 500 | 59.77 | 72.64 | 0.999984 | 0.0035 | 340.05 | 27.66 | Success |
| 1,000 | 57.85 | 70.72 | 0.999976 | 0.0055 | 71.09 | 59.18 | Success |
| 2,000 | 55.22 | 68.10 | 0.999955 | 0.0101 | 198.20 | 39.58 | Success |
| 5,000 | 51.83 | 64.71 | 0.999912 | 0.0220 | 76.24 | 37.48 | Success |
| 10,000 | 49.14 | 62.02 | 0.999846 | 0.0409 | 80.28 | 83.20 | Success |
| 20,000 | 46.20 | 59.08 | 0.999687 | 0.0803 | 69.99 | 29.90 | Success |
| 50,000 | 42.34 | 55.21 | 0.999151 | 0.1958 | 83.06 | 37.89 | Success |
| 100,000 | 39.43 | 52.31 | 0.998314 | 0.3819 | 89.95 | 42.99 | Success |
| 200,000 | 36.41 | 49.29 | 0.996551 | 0.7654 | 73.25 | 33.42 | Success |
| 500,000 | 32.43 | 45.30 | 0.991113 | 1.9170 | 88.84 | 37.78 | Success |
| 1,000,000 | 29.41 | 42.29 | 0.981716 | 3.8376 | 168.12 | 48.37 | Success |

---

## 🌻 8. Performance Plot  

<p align="center">
  <img src="/performance_analysis.png" width="650">
</p>

---

## 🔐 9. Security Highlights  
- AES-CBC → *confidentiality*  
- Arnold scrambling → *chaos & diffusion*  
- SHA-256 → *collision-free embedding map*  
- DWT domain → *compression resilience*  
- High-energy block selection → *imperceptibility*  

---

## 🎨 10. Applications  
- Stealth communication  
- Digital watermarking  
- Medical image confidentiality  
- Forensics  
- Privacy-preserving media sharing  

---

## 🌺 11. Future Add-Ons  
- DWT+DCT hybrid  
- CUDA acceleration  
- JPEG-aware embedding  
- Neural block selection  

---

<div align="center">

## 💖 Crafted by **S. Subhashini**
*A blend of steganography, cryptography, wavelets, chaos theory, and ✨vibes✨.*

</div>
