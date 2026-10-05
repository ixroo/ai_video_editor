import json
import subprocess
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from .downloader import download_media


def inspect_container(video_path: Path) -> Dict[str, Any]:
    """Membaca struktur internal stream video dan audio menggunakan ffprobe."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Berkas video tidak ditemukan: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)

        video_stream = next(
            (s for s in metadata.get("streams", []) if s.get("codec_type") == "video"),
            None
        )
        audio_stream = next(
            (s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"),
            None
        )

        duration = float(metadata.get("format", {}).get("duration", 0.0))
        width = int(video_stream.get("width", 0)) if video_stream else 0
        height = int(video_stream.get("height", 0)) if video_stream else 0
        r_frame_rate = video_stream.get("r_frame_rate", "30/1") if video_stream else "30/1"

        logger.info(
            f"Inspeksi kontainer berhasil: Resolusi={width}x{height}, "
            f"Durasi={duration:.2f}s, Native Framerate={r_frame_rate}"
        )

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "framerate": r_frame_rate,
            "has_audio": audio_stream is not None,
            "raw_metadata": metadata
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Gagal menjalankan ffprobe pada {video_path}: {e.stderr}")
        raise RuntimeError(f"Gagal membaca metadata kontainer: {e.stderr}")


def normalize_to_cfr(
    input_path: Path,
    output_path: Path,
    target_fps: int = 30,
    vaapi_device: str = "/dev/dri/renderD128"
) -> Path:
    """
    Mengonversi video ke Constant Frame Rate (CFR) menggunakan akselerasi AMD VAAPI
    dengan fallback otomatis ke software encoder libx264 jika hardware gagal.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Memulai normalisasi CFR {target_fps}fps untuk: {input_path.name}")

    # Percobaan 1: Menggunakan akselerasi hardware AMD VAAPI
    cmd_vaapi = [
        "ffmpeg", "-y",
        "-vaapi_device", vaapi_device,
        "-hwaccel", "vaapi",
        "-hwaccel_output_format", "vaapi",
        "-i", str(input_path),
        "-vf", "scale_vaapi=format=nv12",
        "-r", str(target_fps),
        "-c:v", "h264_vaapi",
        "-b:v", "6000k",
        "-maxrate", "8000k",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ]

    try:
        logger.info("Mencoba proses transcoding menggunakan hardware akselerasi AMD VAAPI...")
        res = subprocess.run(cmd_vaapi, capture_output=True, text=True)
        if res.returncode == 0:
            logger.success(f"Normalisasi CFR VAAPI berhasil: {output_path.resolve()}")
            return output_path.resolve()
        else:
            logger.warning(f"VAAPI mengembalikan kode non-zero: {res.stderr[:200]}")
    except Exception as e:
        logger.warning(f"Inisialisasi VAAPI gagal: {e}")

    # Percobaan 2: Fallback ke CPU software encoder libx264
    logger.warning("Beralih ke mode fallback: CPU encoder libx264...")
    cmd_cpu = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"fps=fps={target_fps}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ]

    try:
        subprocess.run(cmd_cpu, capture_output=True, text=True, check=True)
        logger.success(f"Normalisasi CFR (CPU Fallback) selesai: {output_path.resolve()}")
        return output_path.resolve()
    except subprocess.CalledProcessError as e:
        logger.error(f"Transcoding CPU gagal: {e.stderr}")
        raise RuntimeError(f"Gagal melakukan normalisasi CFR pada video: {e.stderr}")


def extract_pcm_audio(video_path: Path, output_wav_path: Path) -> Path:
    """
    Mengekstrak stream audio dari kontainer video menjadi berkas WAV mono 16kHz
    sebagai format standar untuk modul Telinga (Whisper & Silero VAD).
    """
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mengekstrak master audio PCM 16kHz mono dari: {video_path.name}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.success(f"Master audio WAV siap: {output_wav_path.resolve()}")
        return output_wav_path.resolve()
    except subprocess.CalledProcessError as e:
        logger.error(f"Gagal mengekstrak audio PCM: {e.stderr}")
        raise RuntimeError(f"Gagal mengekstrak audio: {e.stderr}")


def run_ingest_pipeline(
    source: str,
    workspace_dir: Path,
    fast_audio_only: bool = True
) -> Dict[str, Any]:
    """
    Menjalankan alur Ingest Gateway (Tahap 1):
    1. Unduh atau salin media mentah.
    2. Inspeksi kontainer teknis.
    3. Jika fast_audio_only=True: langsung ekstrak master audio WAV 16kHz mono
       dari video mentah tanpa merender video penuh (hemat waktu ~3 menit).
    4. Jika fast_audio_only=False: jalankan normalisasi CFR penuh (mode legacy).
    """
    workspace = Path(workspace_dir)
    raw_dir = workspace / "input_raw"
    processed_dir = workspace / "normalized"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Langkah 1: Dapatkan berkas media mentah
    downloaded_path = download_media(source, raw_dir)

    # Langkah 2: Inspeksi metadata kontainer teknis
    technical_info = inspect_container(downloaded_path)

    audio_wav_path = processed_dir / f"{downloaded_path.stem}_audio16k.wav"

    if fast_audio_only:
        # Ekstrak audio instan langsung dari video sumber (proses ~2 detik)
        logger.info("Mengaktifkan mode Fast Ingest: Mengekstrak audio langsung tanpa transcode video penuh.")
        master_audio = extract_pcm_audio(downloaded_path, audio_wav_path)
        normalized_video = downloaded_path
    else:
        # Mode lama: Normalisasi CFR 30fps pada seluruh durasi video
        cfr_video_path = processed_dir / f"{downloaded_path.stem}_cfr30.mp4"
        normalized_video = normalize_to_cfr(downloaded_path, cfr_video_path, target_fps=30)
        master_audio = extract_pcm_audio(normalized_video, audio_wav_path)

    manifest = {
        "source_input": source,
        "raw_video_path": str(downloaded_path.resolve()),
        "normalized_video_path": str(normalized_video.resolve()),
        "master_audio_path": str(master_audio.resolve()),
        "technical_info": technical_info,
        "is_fast_ingest": fast_audio_only
    }

    manifest_path = processed_dir / "ingest_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.success(f"Tahap 1 (Ingest Gateway) selesai. Manifes disimpan di {manifest_path.resolve()}")
    return manifest