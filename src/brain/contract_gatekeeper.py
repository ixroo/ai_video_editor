import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Membaca dan memvalidasi file JSON dari disk."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Berkas JSON tidak ditemukan: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_all_words(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mengekstrak seluruh daftar kata dengan stempel waktu dari dokumen Scene Graph."""
    all_words = []
    for sc in scene_graph.get("scenes", []):
        for sentence in sc.get("speech", {}).get("sentences", []):
            for word_obj in sentence.get("words_in_scene", []):
                all_words.append(word_obj)
    # Urutkan berdasarkan waktu mulai kata
    all_words.sort(key=lambda x: x["start"])
    return all_words


def snap_timestamp_to_word_boundary(
    target_time: float,
    words: List[Dict[str, Any]],
    is_start_cut: bool = True,
    pad_buffer: float = 0.05
) -> float:
    """
    Mencegah pemotongan suku kata (Anti-Phoneme Clipping).
    Jika stempel waktu berada di tengah pelafalan kata aktif, geser ke jeda terdekat.
    """
    for w in words:
        w_start = w["start"]
        w_end = w["end"]

        # Periksa apakah titik potong membelah kata ini
        if w_start <= target_time <= w_end:
            if is_start_cut:
                # Geser ke sebelum kata dimulai dengan bantalan jeda
                snapped = max(0.0, w_start - pad_buffer)
                logger.info(
                    f"Anti-phoneme snapping (START): Menggeser {target_time:.3f}s -> {snapped:.3f}s "
                    f"agar tidak memenggal kata '{w['word']}'"
                )
                return round(snapped, 3)
            else:
                # Geser ke setelah kata selesai diucapkan dengan bantalan jeda
                snapped = w_end + pad_buffer
                logger.info(
                    f"Anti-phoneme snapping (END): Menggeser {target_time:.3f}s -> {snapped:.3f}s "
                    f"agar tidak memenggal kata '{w['word']}'"
                )
                return round(snapped, 3)

    return round(target_time, 3)


def clamp_spatial_coordinate(center_x: float, min_val: float = 0.281, max_val: float = 0.719) -> float:
    """
    Memastikan koordinat crop horizontal tetap berada di dalam batas fisik video 16:9
    sehingga tidak menimbulkan garis hitam kosong di tepi kanvas 9:16.
    """
    clamped = max(min_val, min(max_val, center_x))
    if clamped != center_x:
        logger.warning(f"Spatial clamping: Koordinat crop {center_x} dikoreksi menjadi {clamped:.4f}")
    return round(clamped, 4)


def sanitize_editorial_plan(
    draft_plan: Dict[str, Any],
    scene_graph: Dict[str, Any],
    max_allowed_duration: float = 60.0,
    min_clip_duration: float = 0.6
) -> Dict[str, Any]:
    """
    Menjalankan audit kontrak fisik dan deterministik pada draf rencana.
    """
    physical_duration = float(scene_graph.get("duration_seconds", 0.0))
    all_words = extract_all_words(scene_graph)

    raw_clips = draft_plan.get("selected_clips", [])
    sanitized_clips = []
    accumulated_duration = 0.0

    for idx, clip in enumerate(raw_clips):
        raw_start = float(clip.get("source_start", 0.0))
        raw_end = float(clip.get("source_end", 0.0))

        # 1. Batasi titik waktu terhadap durasi media riil (Anti-Halusinasi)
        bound_start = max(0.0, min(physical_duration, raw_start))
        bound_end = max(0.0, min(physical_duration, raw_end))

        # 2. Terapkan anti-phoneme snapping
        snapped_start = snap_timestamp_to_word_boundary(bound_start, all_words, is_start_cut=True)
        snapped_end = snap_timestamp_to_word_boundary(bound_end, all_words, is_start_cut=False)

        # Pastikan start < end
        if snapped_start >= snapped_end:
            snapped_end = min(physical_duration, snapped_start + min_clip_duration)

        clip_dur = snapped_end - snapped_start

        # 3. Buang klip yang terlalu pendek karena rentan memicu glitch visual
        if clip_dur < min_clip_duration:
            logger.warning(
                f"Klip {clip.get('clip_id', idx)} dibuang karena durasi ({clip_dur:.2f}s) "
                f"di bawah batas minimal {min_clip_duration}s"
            )
            continue

        # 4. Batasi akumulasi durasi total agar tidak melebihi batas video pendek
        if accumulated_duration + clip_dur > max_allowed_duration:
            allowed_remain = max_allowed_duration - accumulated_duration
            if allowed_remain >= min_clip_duration:
                snapped_end = snapped_start + allowed_remain
                clip_dur = allowed_remain
            else:
                logger.info(f"Menghentikan penambahan klip: Plafon {max_allowed_duration}s tercapai.")
                break

        # 5. Validasi koordinat spasial
        raw_cx = float(clip.get("optimal_center_x", 0.50))
        validated_cx = clamp_spatial_coordinate(raw_cx)

        sanitized_clips.append({
            "clip_id": f"clip_{len(sanitized_clips) + 1:02d}",
            "scene_id": clip.get("scene_id", f"scene_{idx:03d}"),
            "source_start": round(snapped_start, 3),
            "source_end": round(snapped_end, 3),
            "duration": round(clip_dur, 3),
            "optimal_center_x": validated_cx,
            "reframe_mode": clip.get("reframe_mode", "auto_reframe"),
            "narrative_purpose": clip.get("narrative_purpose", "Dialog terpilih")
        })

        accumulated_duration += clip_dur
        if accumulated_duration >= max_allowed_duration:
            break

    # 6. Validasi segmen hook pembuka
    raw_hook = draft_plan.get("hook_segment", {})
    sanitized_hook = {}
    if sanitized_clips:
        first_clip = sanitized_clips[0]
        hook_start = max(0.0, float(raw_hook.get("source_start", first_clip["source_start"])))
        hook_end = min(
            first_clip["source_end"],
            float(raw_hook.get("source_end", hook_start + min(4.0, first_clip["duration"])))
        )

        sanitized_hook = {
            "source_start": round(hook_start, 3),
            "source_end": round(hook_end, 3),
            "headline": raw_hook.get("headline", "Momen Pembuka Utama"),
            "reason": raw_hook.get("reason", "Segmen aktif awal klip")
        }

    validated_contract = {
        "contract_version": "1.0.0",
        "validation_status": "PASS",
        "target_aspect_ratio": "9:16",
        "physical_media_duration": physical_duration,
        "total_rendered_duration": round(accumulated_duration, 3),
        "total_clips": len(sanitized_clips),
        "hook_segment": sanitized_hook,
        "clips": sanitized_clips,
        "metadata_notes": "Seluruh parameter telah disanitasi dan lolos verifikasi kontrak fisik."
    }

    return validated_contract


def run_contract_gatekeeper_pipeline(
    draft_plan_path: Path,
    scene_graph_path: Path,
    output_dir: Path,
    max_duration: float = 60.0
) -> Dict[str, Any]:
    """
    Menjalankan alur lengkap Gerbang Kontrak (Tahap 6):
    1. Membaca draf rencana editorial dan berkas Scene Graph kanonikal
    2. Menjalankan audit deterministik anti-phoneme clipping
    3. Mengunci koordinat crop visual pada batas aman kanvas
    4. Menerbitkan kontrak validasi final ke validated_plan.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memulai eksekusi Tahap 6 (Gerbang Kontrak Deterministik / Gatekeeper)")

    draft_plan = load_json_file(draft_plan_path)
    scene_graph = load_json_file(scene_graph_path)

    validated_contract = sanitize_editorial_plan(
        draft_plan=draft_plan,
        scene_graph=scene_graph,
        max_allowed_duration=max_duration
    )

    output_path = output_dir / "validated_plan.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validated_contract, f, indent=2, ensure_ascii=False)

    logger.success(f"Tahap 6 selesai. Kontrak eksekusi tersimpan di {output_path.resolve()}")
    return validated_contract