import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .vad_detector import detect_voice_activity
from .transcriber import transcribe_audio_whisper
from .diarization import run_diarization


def run_ear_pipeline(
    audio_wav_path: Path,
    output_dir: Path,
    model_size: str = "base",
    compute_type: str = "int8",
    language: str = "id",
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 400,
    source_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Menjalankan alur Persepsi Auditori (Tahap 2):
    1. Transkripsi kilat via Gemini 3.5 Transcribe di awan.
    2. Melewati Silero VAD dan Diarisasi lokal jika transkripsi awan berhasil (hemat 17 detik).
    3. Ekstraksi jeda hening dan pembicara langsung dari anotasi stempel kata.
    4. Fallback ke Silero VAD dan faster-whisper lokal jika koneksi awan bermasalah.
    """
    audio_wav_path = Path(audio_wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memulai eksekusi Tahap 2 (Ekstraksi Audio / Modul Telinga)")

    # 1. Jalankan transkripsi terlebih dahulu (Utamakan Gemini 3.5 Transcribe di awan)
    transcription_res = transcribe_audio_whisper(
        audio_wav_path=audio_wav_path,
        model_size=model_size,
        compute_type=compute_type,
        language=language,
        source_input=source_input
    )

    segments = transcription_res.get("segments", [])
    total_duration = segments[-1]["end"] if segments else 0.0

    # 2. Cek apakah hasil transkripsi membawa data pembicara bawaan (Gemini Transcribe)
    has_speaker_info = any(
        any("speaker" in w for w in seg.get("words", []))
        for seg in segments
    )

    if has_speaker_info or len(segments) > 0:
        logger.info("Transkripsi awan Gemini berhasil. Melewati Silero VAD dan Diarisasi lokal (Bypass ~17 detik).")
        silence_intervals = []
        speakers_found = set()

        for idx in range(len(segments) - 1):
            gap_start = segments[idx]["end"]
            gap_end = segments[idx + 1]["start"]
            if (gap_end - gap_start) >= 0.35:
                silence_intervals.append({
                    "start": round(gap_start, 3),
                    "end": round(gap_end, 3),
                    "duration": round(gap_end - gap_start, 3)
                })
            for w in segments[idx].get("words", []):
                if "speaker" in w:
                    speakers_found.add(w["speaker"])

        speech_time = sum(s["end"] - s["start"] for s in segments)
        speech_ratio = round(speech_time / max(1.0, total_duration), 3)

        vad_res = {
            "total_duration": total_duration,
            "speech_ratio": speech_ratio,
            "active_speech_segments": len(segments),
            "silence_segments": len(silence_intervals),
            "silence_intervals": silence_intervals
        }
        diarization_res = {
            "speaker_count": max(1, len(speakers_found)),
            "speakers": list(speakers_found) if speakers_found else ["spk_0"],
            "source": "gemini-3.5-transcribe"
        }
    else:
        logger.info("Menjalankan Silero VAD lokal sebagai cadangan...")
        try:
            vad_res = detect_voice_activity(
                audio_wav_path=audio_wav_path,
                threshold=vad_threshold,
                min_silence_duration_ms=vad_min_silence_ms
            )
        except Exception:
            vad_res = detect_voice_activity(audio_wav_path)

        try:
            diarization_res = run_diarization(audio_wav_path)
        except Exception as e:
            logger.warning(f"Diarisasi lokal dilewati ({e}). Menggunakan presenter tunggal.")
            diarization_res = {
                "speaker_count": 1,
                "segments": [{"speaker": "spk_0", "start": 0.0, "end": vad_res.get("total_duration", total_duration)}]
            }

    acoustic_metadata = {
        "audio_path": str(audio_wav_path.resolve()),
        "total_duration": vad_res.get("total_duration", total_duration),
        "speech_ratio": vad_res.get("speech_ratio", 0.0),
        "vad": vad_res,
        "silence_intervals": vad_res.get("silence_intervals", []),
        "transcription": transcription_res,
        "diarization": diarization_res
    }

    output_json = output_dir / "acoustic_metadata.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(acoustic_metadata, f, indent=2, ensure_ascii=False)

    logger.success(f"Tahap 2 Selesai. Hasil disimpan di: {output_json.resolve()}")
    return acoustic_metadata