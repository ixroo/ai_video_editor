import ctypes
import gc
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from faster_whisper import WhisperModel


def release_memory():
    """Memaksa Python dan kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def load_env_file(env_path: Path = Path(".env")) -> None:
    """Membaca berkas .env secara manual tanpa dependensi eksternal."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')
    except Exception as e:
        logger.warning(f"Gagal membaca berkas .env - {e}")


def parse_offset_seconds(val: Any) -> float:
    """Mengonversi nilai offset waktu string atau float ke detik desimal murni."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return round(float(val), 3)
    s = str(val).rstrip("s").strip()
    try:
        return round(float(s), 3)
    except ValueError:
        return 0.0


def extract_word_annotations(interaction: Any) -> List[Any]:
    """Mengekstrak seluruh anotasi tingkat kata dari objek respon Interactions API."""
    words = []
    for step in getattr(interaction, "steps", ()) or []:
        for content in getattr(step, "content", []) or []:
            for annotation in getattr(content, "annotations", ()) or []:
                if getattr(annotation, "type", None) == "word_info":
                    words.append(annotation)
    return words


def transcribe_audio_gemini_api(
    audio_wav_path: Path,
    language_code: str = "id-ID"
) -> Optional[Dict[str, Any]]:
    """
    Mentranskripsi audio ke teks dengan stempel kata presisi via model gemini-3.5-transcribe.
    Menggunakan Interactions API Google untuk memproses audio 23 menit dalam hitungan detik.
    """
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "masukkan_api_key_lo_di_sini":
        logger.warning("GEMINI_API_KEY belum tersedia di .env. Melewati transkripsi Gemini.")
        return None

    from google import genai

    logger.info(f"Mengunggah audio ke Google Files API untuk transkripsi kilat: {audio_wav_path.name}")
    client = genai.Client(api_key=api_key)
    uploaded_file = None

    try:
        # Unggah berkas WAV mono 16kHz ke Files API
        uploaded_file = client.files.upload(file=str(audio_wav_path))
        logger.info(f"Audio berhasil diunggah (URI: {uploaded_file.uri}). Menjalankan model gemini-3.5-transcribe...")

        # Eksekusi transkripsi verbatim dengan word timestamps dan diarization
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[
                {
                    "type": "audio",
                    "uri": uploaded_file.uri,
                    "mime_type": "audio/wav"
                }
            ],
            generation_config={
                "transcription_config": {
                    "language_codes": [language_code],
                    "mode": {
                        "type": "verbatim",
                        "diarization_mode": "speaker",
                        "timestamp_granularities": ["word"]
                    }
                }
            }
        )

        raw_words = extract_word_annotations(interaction)
        if not raw_words:
            logger.warning("Anotasi kata kosong dari respon gemini-3.5-transcribe.")
            return None

        # Susun daftar kata terstruktur
        structured_words = []
        for w in raw_words:
            clean_text = getattr(w, "text", "").strip()
            if not clean_text:
                continue
            start_s = parse_offset_seconds(getattr(w, "start_offset", 0.0))
            end_s = parse_offset_seconds(getattr(w, "end_offset", start_s + 0.2))
            speaker = getattr(w, "speaker", "spk_0")

            structured_words.append({
                "word": clean_text,
                "start": start_s,
                "end": max(end_s, start_s + 0.1),
                "speaker": speaker,
                "probability": 1.0
            })

        if not structured_words:
            return None

        # Kelompokkan kata menjadi segmen kalimat logis berdasarkan jeda nafas atau batas kata
        extracted_segments = []
        full_transcript = []
        current_segment_words = []
        seg_id = 1

        for idx, w in enumerate(structured_words):
            current_segment_words.append(w)
            is_last = idx == len(structured_words) - 1

            pause_after = 0.0
            if not is_last:
                pause_after = structured_words[idx + 1]["start"] - w["end"]

            ends_with_punct = bool(re.search(r"[.!?]$", w["word"]))
            segment_long_enough = len(current_segment_words) >= 12

            if is_last or pause_after >= 0.45 or (segment_long_enough and ends_with_punct):
                seg_text = " ".join(item["word"] for item in current_segment_words)
                seg_start = current_segment_words[0]["start"]
                seg_end = current_segment_words[-1]["end"]

                extracted_segments.append({
                    "id": seg_id,
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                    "words": current_segment_words
                })
                full_transcript.append(seg_text)
                seg_id += 1
                current_segment_words = []

        complete_text = " ".join(full_transcript)
        logger.success(
            f"Transkripsi Gemini 3.5 Transcribe selesai! Terpetakan {len(structured_words)} kata "
            f"dalam {len(extracted_segments)} segmen ucapan presisi tinggi."
        )

        return {
            "language": language_code,
            "language_probability": 1.0,
            "complete_text": complete_text,
            "segments": extracted_segments
        }

    except Exception as e:
        logger.warning(f"Kegagalan pada pemanggilan gemini-3.5-transcribe: {e}")
        return None
    finally:
        # Hapus berkas audio di penyimpanan Google Files API agar tidak menumpuk
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                client.files.delete(name=uploaded_file.name)
                logger.info("Berkas audio sementara di Google Files API berhasil dibersihkan.")
            except Exception:
                pass


def extract_youtube_id(text: str) -> Optional[str]:
    """Mengekstrak 11 karakter YouTube Video ID dari URL atau nama berkas."""
    url_patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$"
    ]
    for p in url_patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)

    stem_name = Path(text).stem
    parts = stem_name.split("_")
    if parts and len(parts[0]) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", parts[0]):
        return parts[0]

    m_free = re.search(r"([a-zA-Z0-9_-]{11})", text)
    if m_free:
        candidate = m_free.group(1)
        if any(c.isdigit() for c in candidate) or "_" in candidate or "-" in candidate:
            return candidate

    return None


def fetch_youtube_subtitles_json3(
    video_id: str,
    temp_dir: Path,
    language: str = "id"
) -> Optional[Dict[str, Any]]:
    """Menyedot subtitle otomatis per kata dari YouTube dalam format JSON3 via yt-dlp."""
    logger.info(f"Mencoba mengambil subtitle native YouTube (ID: {video_id}) format JSON3...")
    out_template = str(temp_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extractor-args", "youtube:player_client=android",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs", f"{language},id,id-orig,en",
        "--sub-format", "json3",
        "--skip-download",
        "-o", out_template,
        f"https://www.youtube.com/watch?v={video_id}"
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return None

        json3_files = list(temp_dir.glob("*.json3"))
        if not json3_files:
            return None

        chosen_file = json3_files[0]
        for jf in json3_files:
            if f".{language}." in jf.name or ".id." in jf.name:
                chosen_file = jf
                break

        with open(chosen_file, "r", encoding="utf-8") as f:
            sub_data = json.load(f)

        events = sub_data.get("events", [])
        if not events:
            return None

        extracted_segments = []
        full_transcript = []
        seg_id = 1

        for ev in events:
            segs = ev.get("segs")
            if not segs:
                continue

            t_start_ms = ev.get("tStartMs", 0)
            d_dur_ms = ev.get("dDurationMs", 0)
            seg_start = round(t_start_ms / 1000.0, 3)
            seg_end = round((t_start_ms + d_dur_ms) / 1000.0, 3)

            tokens = []
            for s in segs:
                raw_w = s.get("utf8", "")
                clean_w = raw_w.replace("\n", "").strip()
                if not clean_w:
                    continue
                offset_ms = s.get("tOffsetMs", 0)
                w_start = round((t_start_ms + offset_ms) / 1000.0, 3)
                tokens.append((clean_w, w_start))

            if not tokens:
                continue

            words_data = []
            for idx, (clean_w, w_start) in enumerate(tokens):
                if idx + 1 < len(tokens):
                    w_end = tokens[idx + 1][1]
                else:
                    w_end = max(w_start + 0.15, seg_end)

                if w_end <= w_start:
                    w_end = round(w_start + 0.15, 3)

                words_data.append({
                    "word": clean_w,
                    "start": w_start,
                    "end": round(w_end, 3),
                    "probability": 1.0
                })

            seg_text = " ".join(w["word"] for w in words_data)
            full_transcript.append(seg_text)
            extracted_segments.append({
                "id": seg_id,
                "start": seg_start,
                "end": max(seg_end, words_data[-1]["end"]),
                "text": seg_text,
                "words": words_data
            })
            seg_id += 1

        if not extracted_segments:
            return None

        complete_text = " ".join(full_transcript)
        logger.success(f"Subtitle YouTube berhasil diimpor! Total {len(extracted_segments)} segmen ucapan.")

        return {
            "language": language,
            "language_probability": 1.0,
            "complete_text": complete_text,
            "segments": extracted_segments
        }

    except Exception:
        return None


def transcribe_audio_whisper(
    audio_wav_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "id",
    source_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orkestrator transkripsi audio multi-jalur:
    1. Utamakan Gemini 3.5 Transcribe via Interactions API (Akurasi tinggi, stempel kata presisi)
    2. Coba penarikan subtitle YouTube jika input memuat ID YouTube
    3. Fallback ke faster-whisper lokal multi-thread jika jaringan atau kuota bermasalah
    """
    audio_path = Path(audio_wav_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

    # Jalur Utama 1: Transkripsi Awan Gemini 3.5 Transcribe
    gemini_result = transcribe_audio_gemini_api(audio_path, language_code="id-ID")
    if gemini_result:
        return gemini_result

    # Jalur Alternatif 2: Sedot Subtitle YouTube
    yt_id = extract_youtube_id(source_input or "") or extract_youtube_id(audio_path.name)
    if yt_id:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yt_subs = fetch_youtube_subtitles_json3(yt_id, Path(tmp_dir), language=language)
            if yt_subs:
                return yt_subs

    # Jalur Cadangan 3: faster-whisper lokal
    available_cores = os.cpu_count() or 4
    optimal_threads = max(4, available_cores)

    logger.info(
        f"Menjalankan faster-whisper lokal sebagai cadangan [{model_size} | {compute_type} | {optimal_threads} Threads] "
        f"untuk: {audio_path.name}"
    )

    initial_prompt = (
        "Halo semua, kita lagi ngobrol santai nih. "
        "Grab, ojol, startup, software, CRM, ESOP, MSOP, scarcity, fragile, dividend, "
        "investasi, aset, cash, market, Bitcoin, kripto, gak, nggak, banget, gimana, kenapa."
    )

    model = None
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=optimal_threads,
            num_workers=1
        )

        segments_generator, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        full_transcript = []
        extracted_segments = []

        for segment in segments_generator:
            full_transcript.append(segment.text.strip())

            words_data = []
            if segment.words:
                for w in segment.words:
                    words_data.append({
                        "word": w.word.strip(),
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "probability": round(float(w.probability), 3)
                    })

            extracted_segments.append({
                "id": segment.id,
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "words": words_data
            })

        complete_text = " ".join(full_transcript)
        logger.success(f"Transkripsi lokal selesai. Dihasilkan {len(extracted_segments)} segmen ucapan.")

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "complete_text": complete_text,
            "segments": extracted_segments
        }

    except Exception as e:
        logger.error(f"Kegagalan saat menjalankan faster-whisper cadangan: {e}")
        raise e
    finally:
        del model
        release_memory()