import os
import ctypes
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


def release_memory():
    """Memaksa Python dan kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def run_diarization(
    audio_wav_path: Path,
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Menjalankan pemisahan pembicara menggunakan pyannote.audio.
    Jika token Hugging Face tidak ditemukan atau proses gagal, sistem otomatis
    mengaktifkan mode fallback presenter tunggal agar alur kerja tidak mandek.
    
    Args:
        audio_wav_path: Path berkas WAV PCM 16kHz mono.
        hf_token: Token otentikasi opsional Hugging Face.
        
    Returns:
        Daftar dictionary rentang waktu giliran bicara per identitas pembicara.
    """
    audio_path = Path(audio_wav_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Berkas audio tidak ditemukan: {audio_path}")

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        logger.warning(
            "Token Hugging Face tidak ditemukan. "
            "Mengaktifkan mode fallback presenter tunggal (spk_0)."
        )
        return [
            {
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 999999.0
            }
        ]

    logger.info(f"Menjalankan pyannote.audio diarization untuk {audio_path.name}")

    pipeline = None
    try:
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )
        
        # Batasi inferensi ke CPU
        pipeline.to(torch.device("cpu"))

        diarization = pipeline(str(audio_path))
        speaker_turns = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_turns.append({
                "speaker_id": str(speaker),
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3)
            })

        logger.success(f"Diarization selesai dengan {len(speaker_turns)} pergantian giliran bicara.")
        return speaker_turns

    except Exception as e:
        logger.warning(f"Pyannote diarization gagal dijalankan ({e}). Mengalihkan ke fallback speaker tunggal.")
        return [
            {
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 999999.0
            }
        ]
    finally:
        del pipeline
        release_memory()