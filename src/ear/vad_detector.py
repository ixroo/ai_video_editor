import json
from pathlib import Path
from typing import Dict, Any, List
import soundfile as sf
import torch
from loguru import logger
from silero_vad import load_silero_vad, get_speech_timestamps


def read_audio_soundfile(audio_path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Membaca berkas audio secara mandiri menggunakan soundfile dan mengubahnya ke PyTorch Tensor
    tanpa memicu torchaudio atau torchcodec yang mencari pustaka CUDA NVIDIA.
    """
    wav_data, sr = sf.read(audio_path, dtype="float32")
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)

    wav_tensor = torch.from_numpy(wav_data)

    if sr != sampling_rate:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav_tensor = resampler(wav_tensor)

    return wav_tensor


def detect_voice_activity(
    audio_path: Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 100
) -> Dict[str, Any]:
    """
    Mendeteksi segmen vokal aktif dan jeda hening menggunakan model Silero VAD
    dengan pembaca audio berbasis soundfile yang aman untuk platform AMD CPU.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Berkas audio tidak ditemukan: {audio_path}")

    logger.info(f"Menjalankan Silero VAD untuk: {audio_path.name}")

    try:
        model = load_silero_vad()
        wav = read_audio_soundfile(str(audio_path), sampling_rate=16000)

        # Mendapatkan stempel waktu segmen vokal dalam satuan detik
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True
        )

        speech_segments: List[Dict[str, float]] = []
        for seg in speech_timestamps:
            start = round(float(seg["start"]), 3)
            end = round(float(seg["end"]), 3)
            duration = round(end - start, 3)
            speech_segments.append({
                "start": start,
                "end": end,
                "duration": duration
            })

        # Menghitung durasi total audio dari panjang tensor sampel
        total_duration = round(len(wav) / 16000.0, 3)

        # Menghitung segmen jeda hening di antara segmen suara
        silence_segments: List[Dict[str, float]] = []
        last_end = 0.0

        for seg in speech_segments:
            gap = round(seg["start"] - last_end, 3)
            if gap > 0.1:
                silence_segments.append({
                    "start": round(last_end, 3),
                    "end": round(seg["start"], 3),
                    "duration": gap
                })
            last_end = seg["end"]

        if last_end < total_duration:
            trailing_gap = round(total_duration - last_end, 3)
            if trailing_gap > 0.1:
                silence_segments.append({
                    "start": round(last_end, 3),
                    "end": total_duration,
                    "duration": trailing_gap
                })

        total_speech_duration = round(sum(s["duration"] for s in speech_segments), 3)
        total_silence_duration = round(sum(s["duration"] for s in silence_segments), 3)
        speech_ratio = round(total_speech_duration / total_duration, 4) if total_duration > 0 else 0.0

        logger.success(
            f"VAD Selesai: Terdeteksi {len(speech_segments)} segmen suara aktif dan "
            f"{len(silence_segments)} segmen jeda hening."
        )

        return {
            "audio_path": str(audio_path.resolve()),
            "total_duration": total_duration,
            "total_speech_duration": total_speech_duration,
            "total_silence_duration": total_silence_duration,
            "speech_ratio": speech_ratio,
            "speech_segments": speech_segments,
            "silence_segments": silence_segments,
            "silence_intervals": silence_segments
        }

    except Exception as e:
        logger.error(f"Kegagalan saat menjalankan Silero VAD: {e}")
        raise e