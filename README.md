# AI Video Editor Pipeline (High-Speed Architecture)

Pipeline otomatisasi penyuntingan video otonom berbasis Python yang dirancang untuk mengonversi rekaman panjang (*long-form podcast/interviews* baik dari berkas lokal maupun unduhan URL langsung) menjadi video pendek vertikal 9:16 (Shorts/Reels/TikTok) secara kilat tanpa memerlukan GPU *dedicated* berdaya besar di lingkungan Linux Mint.

---

## 🏛️ Arsitektur Sistem Modular

Pipeline dirancang menggunakan pola **Decoupled Perception-Reasoning-Execution**:

```
[ Input: Berkas Lokal / URL ] ──► (1) Ingest Gateway (yt-dlp, CFR Normalizer & Audio Demux)
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
    (2) TELINGA (Persepsi Audio)                              (3) MATA (Persepsi Visual)
    - Gemini Cloud Transcribe                                 - MediaPipe Face & Landmark Tracking
    - Millisecond Word Timestamps                             - Center / Adaptive Reframe Geometry
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            ▼
                             (4) Scene Graph Aggregator
                                            ▼
                             (5) OTAK (Cognitive Planner)
                             - Gemini Cascading Fallback (3.7 ➔ 3.6 ➔ 3.5 ➔ 2.5)
                             - 3-Act Narrative Curation (Hook - Body - Conclusion)
                             - Anti-Dangling Sentence Snapper
                                            ▼
                             (6) Contract Gatekeeper (Deterministik)
                                            ▼
                             (7) TANGAN (Mesin Eksekusi Render)
                             - OpenTimelineIO Architecture
                             - ASS Subtitle Generator (Karaoke style)
                             - FFmpeg Native AMD VAAPI Hardware Encoding
                                            ▼
                             (8) Post-Render QC Audit (PASS / FAIL)
                             - Black Frame & Freeze Detector
                             - Headroom Audio Limiting (-1.4 dBFS)
```

### Pilar Utama:

1. **Ingest Gateway:** Mendukung masukan berkas lokal maupun unduhan otomatis dari URL publik (YouTube/podcast) via `yt-dlp`, normalisasi CFR, dan ekstraksi master audio PCM 16kHz.
2. **Modul Telinga (Acoustic Perception):** Mengunggah stream audio PCM mono ke Google Files API untuk transkripsi presisi kata per kata dalam hitungan detik.
3. **Modul Otak (Cognitive Planner):** Dilengkapi sistem *Cascading Fallback*. Jika server Google sibuk (eror 503 / 429), sistem otomatis melompat antar-model gratis (`gemini-3.7-flash` ➔ `gemini-3.6-flash` ➔ `gemini-3.5-flash` ➔ `gemini-3.5-flash-lite` ➔ `gemini-2.5-flash`) secara instan.
4. **Penyaring Anti-Menggantung (Anti-Dangling Snapper):** Algoritma Python deterministik yang memindai kata terakhir klip narasi agar video tidak terpotong pada kata sambung (*"tapi", "karena", "dan", "sehingga", "kalau"*).
5. **Modul Tangan (Rendering Engine):** Memanfaatkan akselerasi perangkat keras Linux VAAPI (`/dev/dri/renderD128`) untuk pemotongan 9:16 dan encoding H.264 di iGPU Radeon Vega, serta perenderan subtitle dinamis via libass.
6. **Quality Control (QC Gate):** Evaluasi otomatis pasca-render untuk mendeteksi frame gelap, kegagalan durasi, dan saturasi audio kliping.

---

## 📋 Kebutuhan Sistem Minimum

* **Sistem Operasi:** Linux Mint (21 / 22), Ubuntu 22.04/24.04 LTS, Debian 12, atau distro Linux modern lainnya.
* **CPU:** Prosesor 64-bit dengan minimal 4 core (Diuji pada AMD Ryzen 3 5300U).
* **RAM:** Minimal 4 GB (Sangat aman di 8 GB berkat manajemen memori sekuensial).
* **Hardware Acceleration:** Driver VAAPI aktif (`/dev/dri/renderD128` - AMD Radeon Mesa radeonsi atau Intel QuickSync).
* **Perangkat Lunak:**
  * Python 3.10+
  * FFmpeg 6.0+ (terkompilasi dengan flag `--enable-vaapi` dan `--enable-libass`)
  * Git & GitHub CLI (`gh`)

---

## 🚀 Panduan Memulai Cepat

### 1. Kloning Repositori

```bash
git clone https://github.com/WhiteTreeDev/ai_video_editor.git
cd ai_video_editor
```

### 2. Buat Lingkungan Virtual (Virtual Environment)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Pasang Dependensi Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Konfigurasi Kunci API

Salin templat berkas `.env.example` menjadi `.env`:

```bash
cp .env.example .env
```

Buka berkas `.env` dan masukkan API Key Gemini Anda:

```env
GEMINI_API_KEY="AIzaSyYourSecretGeminiApiKeyHere"
```

*(Dapatkan kunci gratis di [Google AI Studio](https://aistudio.google.com/)).*

### 5. Verifikasi Akselerasi Hardware VAAPI

Pastikan node render Linux Anda dapat diakses:

```bash
ls -l /dev/dri/renderD128
vainfo
```

---

## 🎬 Menjalankan Pipeline

### Masukan Berkas Lokal:
```bash
python -m src.main --input "path/to/your_podcast_or_video.mp4"
```

### Masukan URL Publik (YouTube/Daring):
```bash
python -m src.main --input "https://www.youtube.com/watch?v=CONTOH_ID"
```

### Struktur Berkas Keluaran:
* **Output Video Vertikal:** `storage/workspace/session_<timestamp>/stage7_hand/final_render.mp4`
* **Laporan Kualitas (QC):** `storage/workspace/session_<timestamp>/stage8_qc/qc_report.json`
* **Manifes Rencana Editorial:** `storage/workspace/session_<timestamp>/stage4_brain/validated_plan.json`

---

## 🔒 Privasi dan Keamanan Berkas

* Berkas `.env` dan direktori kerja `storage/` diisolasi di dalam `.gitignore` sehingga API Key dan berkas video berukuran besar tidak akan terunggah ke Git.
* Berkas audio yang diunggah ke Google Files API selama proses transkripsi otomatis dihapus kembali setelah transkrip diterima.

---

## 📄 Lisensi

Proyek ini didistribusikan di bawah lisensi terbuka [MIT License](LICENSE).