import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import yt_dlp
from loguru import logger


def is_valid_url(url_or_path: str) -> bool:
    """Memeriksa apakah string masukan merupakan URL jaringan yang valid."""
    try:
        parsed = urlparse(url_or_path)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def download_media(url_or_path: str, output_dir: Path) -> Path:
    """
    Mengunduh media dari URL publik menggunakan yt-dlp atau memvalidasi
    keberadaan berkas media lokal.

    Args:
        url_or_path: String URL publik (YouTube, TikTok, dll.) atau path berkas lokal.
        output_dir: Direktori target penyimpanan berkas hasil unduhan.

    Returns:
        Path absolut ke berkas media yang siap diproses.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cek apakah masukan merupakan berkas lokal yang sudah ada
    local_file = Path(url_or_path)
    if local_file.is_file() and local_file.exists():
        logger.info(f"Masukan terverifikasi sebagai berkas lokal: {local_file.resolve()}")
        return local_file.resolve()

    # 2. Validasi format URL jika bukan berkas lokal
    if not is_valid_url(url_or_path):
        raise FileNotFoundError(
            f"Masukan '{url_or_path}' bukan URL valid dan berkas lokal tidak ditemukan."
        )

    logger.info(f"Memulai pengunduhan media dari URL: {url_or_path}")
    output_template = str(output_dir / "%(id)s_%(title).50s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoRemuxer",
                "preferedformat": "mp4",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_or_path, download=True)
            candidate_filename = Path(ydl.prepare_filename(info))

            # Pastikan ekstensi mp4 setelah konversi
            final_mp4 = candidate_filename.with_suffix(".mp4")
            if final_mp4.exists():
                logger.success(f"Unduhan selesai dan terverifikasi: {final_mp4.resolve()}")
                return final_mp4.resolve()

            if candidate_filename.exists():
                logger.success(f"Unduhan selesai dan terverifikasi: {candidate_filename.resolve()}")
                return candidate_filename.resolve()

            # Pencarian toleran berdasarkan ID berkas
            media_id = info.get("id")
            matches = list(output_dir.glob(f"{media_id}*"))
            if matches:
                logger.success(f"Berkas terdeteksi melalui ID media: {matches[0].resolve()}")
                return matches[0].resolve()

            raise FileNotFoundError(f"Berkas hasil unduhan {url_or_path} tidak ditemukan di {output_dir}")
    except Exception as e:
        logger.error(f"Kegagalan saat mengunduh media dari {url_or_path}: {e}")
        raise e