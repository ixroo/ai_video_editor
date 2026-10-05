import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Membaca dan memvalidasi berkas JSON dari disk."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Berkas JSON tidak ditemukan: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def seconds_to_frames(seconds: float, fps: int = 30) -> int:
    """Mengonversi detik ke nomor frame bulat pada framerate konstan."""
    return int(round(float(seconds) * fps))


def frames_to_seconds(frames: int, fps: int = 30) -> float:
    """Mengonversi nomor frame kembali ke detik (presisi mikrodetik)."""
    return round(frames / fps, 6)


def compute_frame_ranges(
    validated_plan: Dict[str, Any],
    fps: int = 30,
    max_source_frames: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mengubah klip di validated_plan.json menjadi rentang frame bulat dan
    menghitung posisi tiap klip di garis waktu hasil edit (tanpa jeda).

    PENTING: fungsi ini dipakai bersama oleh subtitle_generator.py agar teks
    dan gambar memakai pembulatan frame yang sama persis (tidak ada geser).
    Titik potong dibulatkan satu per satu (bukan durasi dijumlahkan) supaya
    kesalahan pembulatan tidak menumpuk antar klip.
    """
    ranges: List[Dict[str, Any]] = []
    timeline_cursor = 0

    for idx, clip in enumerate(validated_plan.get("clips", [])):
        start_f = max(0, seconds_to_frames(clip["source_start"], fps))
        end_f = seconds_to_frames(clip["source_end"], fps)

        if max_source_frames is not None:
            end_f = min(end_f, max_source_frames)

        dur_f = end_f - start_f
        if dur_f < 1:
            logger.warning(
                f"Klip {clip.get('clip_id', idx)} dilewati: durasi < 1 frame setelah pembulatan."
            )
            continue

        ranges.append({
            "index": len(ranges) + 1,
            "clip_id": clip.get("clip_id", f"clip_{idx + 1:02d}"),
            "scene_id": clip.get("scene_id", ""),
            "source_start_frame": start_f,
            "source_end_frame": end_f,
            "duration_frames": dur_f,
            "timeline_start_frame": timeline_cursor,
            "optimal_center_x": float(clip.get("optimal_center_x", 0.5)),
            "reframe_mode": clip.get("reframe_mode", "auto_reframe"),
            "narrative_purpose": clip.get("narrative_purpose", ""),
        })
        timeline_cursor += dur_f

    return ranges


def probe_video_size(video_path: Path) -> Tuple[int, int]:
    """Membaca lebar dan tinggi video lewat ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream = json.loads(res.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception as e:
        raise RuntimeError(f"Gagal membaca resolusi video {video_path}: {e}")


def compute_crop_window(
    src_width: int,
    src_height: int,
    center_x: float,
    target_aspect: Tuple[int, int] = (9, 16)
) -> Dict[str, int]:
    """
    Menghitung jendela crop 9:16 dalam piksel dari koordinat pusat ternormalisasi
    (0.0 - 1.0). Hasil selalu genap dan dijamin tidak keluar dari batas video.
    """
    ar_w, ar_h = target_aspect

    # Ambil tinggi penuh; lebar mengikuti rasio target
    crop_h = src_height
    crop_w = int(round(crop_h * ar_w / ar_h))

    # Jika sumber lebih sempit dari 9:16, ambil lebar penuh dan potong tingginya
    if crop_w > src_width:
        crop_w = src_width
        crop_h = int(round(crop_w * ar_h / ar_w))

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    x = int(round(center_x * src_width - crop_w / 2))
    x = max(0, min(src_width - crop_w, x))
    x -= x % 2

    y = (src_height - crop_h) // 2
    y -= y % 2

    return {"w": crop_w, "h": crop_h, "x": x, "y": y}


def build_otio_timeline(
    segments: List[Dict[str, Any]],
    video_path: Path,
    total_source_frames: int,
    fps: int = 30,
    timeline_name: str = "ai_video_editor_timeline"
):
    """
    Membangun timeline OpenTimelineIO dengan satu track video dan satu track audio
    yang berisi klip identik (gambar dan suara terkunci pada titik potong yang sama).
    """
    # Impor ditunda agar modul lain yang hanya butuh hitungan frame tidak wajib memuat OTIO
    import opentimelineio as otio

    rate = float(fps)
    timeline = otio.schema.Timeline(name=timeline_name)
    timeline.global_start_time = otio.opentime.RationalTime(0, rate)

    video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)

    def make_clip(seg: Dict[str, Any], suffix: str):
        media_ref = otio.schema.ExternalReference(
            target_url=str(Path(video_path).resolve()),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(total_source_frames, rate)
            )
        )
        clip = otio.schema.Clip(
            name=f"{seg['clip_id']}_{suffix}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(seg["source_start_frame"], rate),
                duration=otio.opentime.RationalTime(seg["duration_frames"], rate)
            )
        )
        clip.metadata["ai_video_editor"] = {
            "scene_id": seg["scene_id"],
            "optimal_center_x": seg["optimal_center_x"],
            "reframe_mode": seg["reframe_mode"],
            "narrative_purpose": seg["narrative_purpose"],
            "crop": seg["crop"]
        }
        return clip

    for seg in segments:
        video_track.append(make_clip(seg, "V"))
        audio_track.append(make_clip(seg, "A"))

    timeline.tracks.append(video_track)
    timeline.tracks.append(audio_track)
    return timeline


def run_otio_compiler_pipeline(
    validated_plan_path: Path,
    video_path: Path,
    output_dir: Path,
    scene_graph_path: Optional[Path] = None,
    fps: int = 30,
    output_width: int = 1080,
    output_height: int = 1920
) -> Dict[str, Any]:
    """
    Menjalankan kompilasi garis waktu (Tahap 7 bagian 1):
    1. Membaca validated_plan.json
    2. Menghitung titik potong presisi frame pada CFR 30 fps
    3. Menghitung jendela crop 9:16 per klip dari koordinat wajah
    4. Menyusun timeline OTIO (timeline.otio)
    5. Menerbitkan manifes instruksi render (render_manifest.json)

    Args:
        validated_plan_path: Path validated_plan.json hasil Tahap 6.
        video_path: Path video CFR hasil normalisasi Tahap 1.
        output_dir: Direktori penyimpanan timeline.otio dan render_manifest.json.
        scene_graph_path: Path opsional scene_graph.json (cadangan info resolusi).
        fps: Framerate konstan video (harus sama dengan hasil Tahap 1).
        output_width: Lebar video akhir.
        output_height: Tinggi video akhir.

    Returns:
        Dictionary manifes render yang siap dipakai ffmpeg_vaapi.py.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Berkas video tidak ditemukan: {video_path}")

    logger.info("Memulai eksekusi Tahap 7A (Kompilasi Timeline OTIO / Modul Tangan)")

    plan = load_json_file(validated_plan_path)

    # Resolusi dibaca dari video yang benar-benar akan di-crop
    try:
        src_w, src_h = probe_video_size(video_path)
    except RuntimeError:
        if not scene_graph_path:
            raise
        tech = load_json_file(scene_graph_path).get("technical_info", {})
        src_w, src_h = int(tech.get("width", 0)), int(tech.get("height", 0))
        if src_w <= 0 or src_h <= 0:
            raise
        logger.warning("ffprobe gagal, memakai resolusi dari scene_graph.json.")

    # Batas frame fisik video agar titik potong tidak melampaui akhir berkas
    duration_hint = float(plan.get("physical_media_duration", 0.0))
    max_frames = seconds_to_frames(duration_hint, fps) if duration_hint > 0 else None

    segments = compute_frame_ranges(plan, fps=fps, max_source_frames=max_frames)
    if not segments:
        raise ValueError("validated_plan.json tidak memiliki klip yang valid untuk dirender.")

    for seg in segments:
        seg["crop"] = compute_crop_window(src_w, src_h, seg["optimal_center_x"])
        seg["source_start_sec"] = frames_to_seconds(seg["source_start_frame"], fps)
        seg["duration_sec"] = frames_to_seconds(seg["duration_frames"], fps)
        seg["timeline_start_sec"] = frames_to_seconds(seg["timeline_start_frame"], fps)

    total_frames = sum(s["duration_frames"] for s in segments)
    total_source_frames = max_frames if max_frames else segments[-1]["source_end_frame"]
    total_source_frames = max(total_source_frames, max(s["source_end_frame"] for s in segments))

    # Susun dan simpan timeline OTIO
    timeline = build_otio_timeline(segments, video_path, total_source_frames, fps=fps)
    otio_path = output_dir / "timeline.otio"
    import opentimelineio as otio
    otio.adapters.write_to_file(timeline, str(otio_path))

    manifest = {
        "manifest_version": "1.0.0",
        "fps": fps,
        "source_video": str(video_path.resolve()),
        "source_width": src_w,
        "source_height": src_h,
        "output_width": output_width,
        "output_height": output_height,
        "total_frames": total_frames,
        "total_duration": frames_to_seconds(total_frames, fps),
        "otio_path": str(otio_path.resolve()),
        "segments": segments
    }

    manifest_path = output_dir / "render_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Tahap 7A selesai. {len(segments)} klip, {total_frames} frame "
        f"({manifest['total_duration']}s). Manifes: {manifest_path.resolve()}"
    )
    return manifest