import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from .otio_compiler import compute_frame_ranges, frames_to_seconds, load_json_file

# Warna ASS memakai urutan &HAABBGGRR (BGR, bukan RGB)
COLOUR_WHITE = "&H00FFFFFF"
COLOUR_YELLOW = "&H0000FFFF"
COLOUR_BLACK = "&H00000000"
COLOUR_SHADOW = "&H80000000"


def format_ass_time(seconds: float) -> str:
    """Mengonversi detik float ke format waktu subtitle (H:MM:SS.cs)."""
    total_cs = max(0, int(round(seconds * 100)))
    hrs = total_cs // 360000
    mins = (total_cs // 6000) % 60
    secs = (total_cs // 100) % 60
    centis = total_cs % 100
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def sanitize_ass_text(text: str, uppercase: bool = False) -> str:
    """Membersihkan karakter yang bisa dibaca sebagai kode override oleh renderer."""
    cleaned = text.replace("\\", "").replace("{", "(").replace("}", ")")
    cleaned = cleaned.replace("\n", " ").replace("\r", " ").strip()
    return cleaned.upper() if uppercase else cleaned


def clean_token_for_diff(text: str) -> str:
    """Menghapus tanda baca dan mengubah ke huruf kecil untuk perbandingan kata."""
    return re.sub(r"[^\w]", "", text).lower()


def collect_unique_words(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Mengumpulkan seluruh kata dari Scene Graph dengan dukungan multijalur:
    1. Dari struktur scenes -> speech -> sentences -> words_in_scene
    2. Dari scenes -> speech -> segments -> words
    3. Dari root transcription -> segments -> words (Whisper mentah)
    """
    seen = set()
    words: List[Dict[str, Any]] = []

    # Jalur 1 & 2: Dari metadata scenes
    for sc in scene_graph.get("scenes", []):
        speech = sc.get("speech", {})
        for sent in speech.get("sentences", []):
            for w in sent.get("words_in_scene", []):
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                try:
                    w_start = float(w["start"])
                    w_end = float(w["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (round(w_start, 3), round(w_end, 3), text)
                if key not in seen:
                    seen.add(key)
                    words.append({"word": text, "start": w_start, "end": w_end})

        for seg in speech.get("segments", []):
            for w in seg.get("words", []):
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                try:
                    w_start = float(w["start"])
                    w_end = float(w["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (round(w_start, 3), round(w_end, 3), text)
                if key not in seen:
                    seen.add(key)
                    words.append({"word": text, "start": w_start, "end": w_end})

    # Jalur 3: Dari root transcription
    root_segs = (
        scene_graph.get("transcription", {}).get("segments", [])
        or scene_graph.get("acoustic_metadata", {}).get("transcription", {}).get("segments", [])
    )
    for seg in root_segs:
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            try:
                w_start = float(w["start"])
                w_end = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (round(w_start, 3), round(w_end, 3), text)
            if key not in seen:
                seen.add(key)
                words.append({"word": text, "start": w_start, "end": w_end})

    words.sort(key=lambda x: (x["start"], x["end"]))
    return words


def extract_timeline_words(
    scene_graph: Dict[str, Any],
    segments: List[Dict[str, Any]],
    fps: int = 30,
    min_word_duration: float = 0.03
) -> List[Dict[str, Any]]:
    """
    Memindahkan stempel waktu kata dari video mentah ke garis waktu hasil edit.
    """
    raw_words = collect_unique_words(scene_graph)
    timeline_words: List[Dict[str, Any]] = []

    for seg in segments:
        src_start = frames_to_seconds(seg["source_start_frame"], fps)
        src_end = frames_to_seconds(seg["source_end_frame"], fps)
        tl_start = frames_to_seconds(seg["timeline_start_frame"], fps)
        tl_end = tl_start + frames_to_seconds(seg["duration_frames"], fps)

        for w in raw_words:
            midpoint = (w["start"] + w["end"]) / 2.0
            if not (src_start <= midpoint < src_end):
                continue

            rel_start = tl_start + (max(w["start"], src_start) - src_start)
            rel_end = tl_start + (min(w["end"], src_end) - src_start)

            if rel_end - rel_start < min_word_duration:
                rel_end = rel_start + min_word_duration
            rel_end = min(rel_end, tl_end)
            if rel_end <= rel_start:
                continue

            timeline_words.append({
                "word": w["word"],
                "timeline_start": round(rel_start, 3),
                "timeline_end": round(rel_end, 3),
                "clip_index": seg["index"]
            })

    timeline_words.sort(key=lambda x: x["timeline_start"])
    return timeline_words


def align_cleaned_words_with_timestamps(
    raw_timeline_words: List[Dict[str, Any]],
    cleaned_text: str
) -> List[Dict[str, Any]]:
    """
    Menyelaraskan teks bersih bebas typo dari Gemini dengan stempel waktu
    asli per kata dari Whisper menggunakan sequence alignment.
    """
    clean_words = [w for w in cleaned_text.strip().split() if w]
    if not clean_words or not raw_timeline_words:
        return raw_timeline_words

    raw_tokens = [w["word"] for w in raw_timeline_words]
    norm_raw = [clean_token_for_diff(w) for w in raw_tokens]
    norm_clean = [clean_token_for_diff(w) for w in clean_words]

    matcher = difflib.SequenceMatcher(None, norm_raw, norm_clean)
    if matcher.ratio() < 0.35:
        logger.warning(
            f"Kemiripan teks Gemini dengan Whisper terlalu rendah ({matcher.ratio():.2f}). "
            "Menggunakan transkrip asli demi menjaga sinkronisasi."
        )
        return raw_timeline_words

    aligned: List[Dict[str, Any]] = []
    default_clip_idx = raw_timeline_words[0].get("clip_index", 0)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                raw_w = raw_timeline_words[i1 + k]
                aligned.append({
                    "word": clean_words[j1 + k],
                    "timeline_start": raw_w["timeline_start"],
                    "timeline_end": raw_w["timeline_end"],
                    "clip_index": raw_w.get("clip_index", default_clip_idx)
                })
        elif tag == "replace":
            span_start = raw_timeline_words[i1]["timeline_start"]
            span_end = raw_timeline_words[i2 - 1]["timeline_end"]
            span_dur = max(0.05, span_end - span_start)
            num_clean = j2 - j1
            clip_idx = raw_timeline_words[i1].get("clip_index", default_clip_idx)
            step = span_dur / num_clean

            for k in range(num_clean):
                w_start = round(span_start + (k * step), 3)
                w_end = round(span_start + ((k + 1) * step), 3)
                aligned.append({
                    "word": clean_words[j1 + k],
                    "timeline_start": w_start,
                    "timeline_end": w_end,
                    "clip_index": clip_idx
                })
        elif tag == "insert":
            anchor_time = aligned[-1]["timeline_end"] if aligned else (raw_timeline_words[i1]["timeline_start"] if i1 < len(raw_timeline_words) else 0.0)
            clip_idx = aligned[-1].get("clip_index", default_clip_idx) if aligned else default_clip_idx
            num_clean = j2 - j1

            for k in range(num_clean):
                w_start = round(anchor_time + (k * 0.15), 3)
                w_end = round(w_start + 0.15, 3)
                aligned.append({
                    "word": clean_words[j1 + k],
                    "timeline_start": w_start,
                    "timeline_end": w_end,
                    "clip_index": clip_idx
                })
        elif tag == "delete":
            continue

    return aligned if aligned else raw_timeline_words


def group_words_into_lines(
    timeline_words: List[Dict[str, Any]],
    words_per_line: int = 3,
    max_chars_per_line: int = 22,
    max_gap_seconds: float = 0.5
) -> List[List[Dict[str, Any]]]:
    """
    Memecah kata menjadi baris pendek untuk konsumsi video vertikal.
    """
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for w in timeline_words:
        if current:
            prev = current[-1]
            current_chars = sum(len(x["word"]) for x in current) + len(current) - 1
            would_be_chars = current_chars + 1 + len(w["word"])

            break_line = (
                len(current) >= words_per_line
                or would_be_chars > max_chars_per_line
                or (w["timeline_start"] - prev["timeline_end"]) > max_gap_seconds
                or w["clip_index"] != prev["clip_index"]
                or prev["word"].endswith((".", "?", "!", "…"))
                or (prev["word"].endswith(",") and len(current) >= 2)
            )
            if break_line:
                lines.append(current)
                current = []

        current.append(w)

    if current:
        lines.append(current)

    return lines


def build_subtitle_header(
    play_res_x: int,
    play_res_y: int,
    font_name: str,
    font_size: int,
    outline_size: int,
    margin_bottom: int,
    margin_side: int
) -> List[str]:
    """Menyusun header naskah subtitle format ASS."""
    return [
        "[Script Info]",
        "Title: AI Video Editor Karaoke Subtitles",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Karaoke,{font_name},{font_size},{COLOUR_WHITE},{COLOUR_YELLOW},{COLOUR_BLACK},"
        f"{COLOUR_SHADOW},-1,0,0,0,100,100,0,0,1,{outline_size},0,2,"
        f"{margin_side},{margin_side},{margin_bottom},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]


def build_line_events(
    line: List[Dict[str, Any]],
    next_line_start: Optional[float],
    highlight_colour: str,
    base_colour: str,
    uppercase: bool,
    tail_hold: float
) -> List[str]:
    """
    Membuat event highlight kata aktif secara presisi.
    """
    events: List[str] = []
    texts = [sanitize_ass_text(w["word"], uppercase) for w in line]

    for i, w in enumerate(line):
        ev_start = w["timeline_start"]
        if i < len(line) - 1:
            ev_end = line[i + 1]["timeline_start"]
        else:
            ev_end = w["timeline_end"] + tail_hold
            if next_line_start is not None:
                ev_end = min(ev_end, next_line_start)

        if ev_end <= ev_start:
            ev_end = ev_start + 0.05

        hl_bgr = highlight_colour[-6:]
        base_bgr = base_colour[-6:]
        parts = []
        for j, txt in enumerate(texts):
            if j == i:
                parts.append(f"{{\\c&H{hl_bgr}&}}{txt}{{\\c&H{base_bgr}&}}")
            else:
                parts.append(txt)
        text = " ".join(parts)

        events.append(
            f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},"
            f"Karaoke,,0,0,0,,{text}"
        )

    return events


def generate_karaoke_ass(
    timeline_words: List[Dict[str, Any]],
    output_ass_path: Path,
    words_per_line: int = 3,
    max_chars_per_line: int = 22,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "DejaVu Sans",
    font_size: int = 72,
    outline_size: int = 6,
    margin_bottom: int = 460,
    margin_side: int = 60,
    uppercase: bool = False,
    tail_hold: float = 0.15
) -> Dict[str, Any]:
    """
    Menyusun naskah subtitle dengan highlight karaoke di area aman vertikal 9:16.
    """
    output_ass_path = Path(output_ass_path)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    lines_data = group_words_into_lines(
        timeline_words,
        words_per_line=words_per_line,
        max_chars_per_line=max_chars_per_line
    )

    doc = build_subtitle_header(
        play_res_x, play_res_y, font_name, font_size,
        outline_size, margin_bottom, margin_side
    )

    for idx, line in enumerate(lines_data):
        next_start = lines_data[idx + 1][0]["timeline_start"] if idx + 1 < len(lines_data) else None
        doc.extend(
            build_line_events(
                line, next_start, COLOUR_YELLOW, COLOUR_WHITE, uppercase, tail_hold
            )
        )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc) + "\n")

    return {
        "subtitle_path": str(output_ass_path.resolve()),
        "total_words": len(timeline_words),
        "total_lines": len(lines_data)
    }


def run_subtitle_pipeline(
    scene_graph_path: Path,
    validated_plan_path: Path,
    output_dir: Path,
    fps: int = 30,
    words_per_line: int = 3,
    max_chars_per_line: int = 22,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "DejaVu Sans",
    font_size: int = 72,
    margin_bottom: int = 460,
    uppercase: bool = False
) -> Dict[str, Any]:
    """
    Menjalankan pembuatan subtitle (Tahap 7B):
    1. Membaca kata transkripsi dari scene_graph.json
    2. Memetakan kata ke timeline klip di validated_plan.json
    3. Menyelaraskan teks ejaan rapi buatan Gemini ke stempel waktu fisik Whisper
    4. Menghasilkan subtitle.ass dengan animasi karaoke
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memulai eksekusi Tahap 7B (Pembuatan Subtitle / Modul Tangan)")

    scene_graph = load_json_file(scene_graph_path)
    plan = load_json_file(validated_plan_path)

    duration_hint = float(plan.get("physical_media_duration", 0.0))
    max_frames = int(duration_hint * fps) if duration_hint > 0 else None
    segments = compute_frame_ranges(plan, fps=fps, max_source_frames=max_frames)

    timeline_words = extract_timeline_words(scene_graph, segments, fps=fps)

    cleaned_transcript = plan.get("cleaned_transcript", "").strip()
    if not cleaned_transcript and plan.get("selected_clips"):
        cleaned_transcript = plan["selected_clips"][0].get("cleaned_transcript", "").strip()

    if cleaned_transcript and timeline_words:
        logger.info("Menyelaraskan teks bersih bebas typo dari Gemini dengan stempel waktu Whisper...")
        timeline_words = align_cleaned_words_with_timestamps(timeline_words, cleaned_transcript)

    if not timeline_words:
        logger.warning("Tidak ada kata ucapan pada klip terpilih. Subtitle akan kosong.")

    words_path = output_dir / "timeline_words.json"
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(timeline_words, f, indent=2, ensure_ascii=False)

    ass_path = output_dir / "subtitles.ass"
    result = generate_karaoke_ass(
        timeline_words,
        ass_path,
        words_per_line=words_per_line,
        max_chars_per_line=max_chars_per_line,
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        font_name=font_name,
        font_size=font_size,
        margin_bottom=margin_bottom,
        uppercase=uppercase
    )
    result["timeline_words_path"] = str(words_path.resolve())

    logger.success(
        f"Tahap 7B selesai. {result['total_words']} kata dalam {result['total_lines']} baris. "
        f"Subtitle: {result['subtitle_path']}"
    )
    return result