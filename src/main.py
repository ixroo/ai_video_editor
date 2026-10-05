import argparse
import ctypes
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from src.ingest.normalizer import run_ingest_pipeline
from src.ear import run_ear_pipeline
from src.eye import run_eye_pipeline
from src.brain import run_brain_pipeline
from src.brain.contract_gatekeeper import run_contract_gatekeeper_pipeline
from src.hand import run_hand_pipeline
from src.qc import run_post_render_qc


def release_memory() -> None:
    """Memaksa Python dan runtime kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def load_config(config_path: Path) -> Dict[str, Any]:
    """Membaca berkas konfigurasi sistem pipeline_config.json."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_session_workspace(base_dir: Path) -> Path:
    """Membuat direktori kerja terisolasi berbasis stempel waktu."""
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = base_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def parse_arguments() -> argparse.Namespace:
    """Mengatur antarmuka baris perintah CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline Otomatisasi Video Editor Vertikal AI (High-Speed YouTube Optimized)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path berkas video lokal atau URL publik (YouTube, TikTok, dll.)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/pipeline_config.json",
        help="Jalur berkas konfigurasi JSON (default: config/pipeline_config.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Nama model Gemini AI (opsional, default membaca dari pipeline_config.json)"
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Menonaktifkan pembakaran subtitle pada video akhir"
    )
    parser.add_argument(
        "--no-vaapi",
        action="store_true",
        help="Memaksa proses transcoding menggunakan CPU software encoder libx264"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    start_total_time = time.time()

    logger.info("Memulai inisialisasi AI Video Editor Pipeline (High-Speed Architecture)")
    config = load_config(Path(args.config))

    general_cfg = config.get("general", {})
    hardware_cfg = config.get("hardware", {})
    ear_cfg = config.get("ear", {})
    eye_cfg = config.get("eye", {})
    brain_cfg = config.get("brain", {})
    hand_cfg = config.get("hand", {})

    workspace_base = Path(general_cfg.get("workspace_dir", "storage/workspace"))
    session_workspace = setup_session_workspace(workspace_base)
    logger.info(f"Ruang kerja sesi aktif: {session_workspace.resolve()}")

    target_fps = int(general_cfg.get("target_fps", 30))
    vaapi_device = hardware_cfg.get("vaapi_device", "/dev/dri/renderD128")
    use_hardware = hardware_cfg.get("use_vaapi", True) and not args.no_vaapi
    gemini_model = args.model or brain_cfg.get("gemini_model", "gemini-3.7-flash")
    fallback_model = brain_cfg.get("gemini_fallback_model", "gemini-3.7-flash")
    max_duration = float(brain_cfg.get("max_allowed_duration_seconds", 60.0))
    enable_subtitles = hand_cfg.get("enable_subtitles", True) and not args.no_subtitles

    whisper_model = ear_cfg.get("whisper_model_size", "base")
    whisper_compute = ear_cfg.get("whisper_compute_type", "int8")
    whisper_lang = ear_cfg.get("whisper_language", "id")

    # 1. Tahap 1: Fast Ingest Gateway (Ekstraksi audio WAV instan ~2 detik)
    logger.info("Eksekusi Tahap 1: Fast Ingest Gateway")
    ingest_dir = session_workspace / "stage1_ingest"
    ingest_manifest = run_ingest_pipeline(source=args.input, workspace_dir=ingest_dir, fast_audio_only=True)
    raw_video = Path(ingest_manifest["raw_video_path"])
    master_audio = Path(ingest_manifest["master_audio_path"])
    ingest_manifest_path = ingest_dir / "normalized" / "ingest_manifest.json"
    release_memory()

    # 2. Tahap 2: Ekstraksi Audio & Transkripsi (Gemini Transcribe First)
    logger.info("Eksekusi Tahap 2: Ekstraksi Audio & Transkripsi (Awan Gemini 3.5 Transcribe)")
    ear_dir = session_workspace / "stage2_ear"
    acoustic_metadata = run_ear_pipeline(
        audio_wav_path=master_audio,
        output_dir=ear_dir,
        model_size=whisper_model,
        compute_type=whisper_compute,
        language=whisper_lang,
        vad_threshold=float(ear_cfg.get("vad_threshold", 0.5)),
        vad_min_silence_ms=int(ear_cfg.get("vad_min_silence_duration_ms", 400)),
        source_input=args.input
    )
    acoustic_json_path = ear_dir / "acoustic_metadata.json"
    release_memory()

    # 3. Tahap 4-6 (Fase Awal): Perencanaan Narasi AI Berbasis Audio
    logger.info(f"Eksekusi Tahap 4-6 (Fase Awal): Perencanaan Narasi AI dengan Model {gemini_model}")
    brain_dir = session_workspace / "stage4_brain"
    initial_plan = run_brain_pipeline(
        acoustic_json_path=acoustic_json_path,
        visual_json_path=None,
        output_dir=brain_dir,
        manifest_json_path=ingest_manifest_path,
        model_name=gemini_model,
        fallback_model=fallback_model,
        target_max_duration=max_duration
    )
    draft_plan_path = brain_dir / "draft_plan.json"
    scene_graph_path = brain_dir / "scene_graph.json"

    with open(draft_plan_path, "r", encoding="utf-8") as f:
        draft_plan_data = json.load(f)

    clips = draft_plan_data.get("selected_clips", []) or initial_plan.get("segments", []) or initial_plan.get("selected_clips", [])
    if not clips:
        raise ValueError("Tidak ada segmen narasi yang berhasil dipilih oleh sistem perencana AI.")

    first_clip = clips[0]
    target_start = float(first_clip.get("source_start", 0.0))
    target_end = float(first_clip.get("source_end", target_start + 30.0))
    logger.info(f"Klip narasi terpilih oleh Gemini: {target_start:.2f}s - {target_end:.2f}s")
    release_memory()

    # 4. Tahap 3: Ekstraksi Visual Selektif
    logger.info("Eksekusi Tahap 3: Pelacakan Wajah Selektif MediaPipe")
    eye_dir = session_workspace / "stage3_eye"
    visual_metadata = run_eye_pipeline(
        video_path=raw_video,
        output_dir=eye_dir,
        scene_detect_enabled=False,
        face_track_enabled=eye_cfg.get("face_tracking_enabled", False),
        ocr_enabled=False,
        target_start=target_start,
        target_end=target_end
    )
    release_memory()

    sc_reframe = visual_metadata["scenes"][0].get("reframe", {})
    optimal_cx = sc_reframe.get("optimal_center_x", 0.50)
    reframe_mode = sc_reframe.get("mode", "static_center")

    if "selected_clips" in draft_plan_data and draft_plan_data["selected_clips"]:
        draft_plan_data["selected_clips"][0]["optimal_center_x"] = optimal_cx
        draft_plan_data["selected_clips"][0]["reframe_mode"] = reframe_mode
    with open(draft_plan_path, "w", encoding="utf-8") as f:
        json.dump(draft_plan_data, f, indent=2, ensure_ascii=False)

    validated_plan = run_contract_gatekeeper_pipeline(
        draft_plan_path=draft_plan_path,
        scene_graph_path=scene_graph_path,
        output_dir=brain_dir,
        max_duration=max_duration
    )
    validated_plan_path = brain_dir / "validated_plan.json"
    release_memory()

    # 5. Tahap 7: Eksekusi dan Render Fisik
    logger.info("Eksekusi Tahap 7: Kompilasi dan Render Fisik Terakselerasi VAAPI")
    hand_dir = session_workspace / "stage7_hand"
    render_result = run_hand_pipeline(
        validated_plan_path=validated_plan_path,
        scene_graph_path=scene_graph_path,
        output_dir=hand_dir,
        video_path=raw_video,
        fps=target_fps,
        enable_subtitles=enable_subtitles,
        use_hardware=use_hardware,
        vaapi_device=vaapi_device,
        words_per_line=hand_cfg.get("subtitle_words_per_line", 3),
        font_name=hand_cfg.get("subtitle_font_name", "DejaVu Sans"),
        font_size=hand_cfg.get("subtitle_font_size", 72),
        margin_bottom=hand_cfg.get("subtitle_margin_bottom", 460),
        uppercase=hand_cfg.get("subtitle_uppercase", False),
        keep_temp=hand_cfg.get("keep_temp_segments", False)
    )
    rendered_video_path = Path(render_result["output_path"])
    render_manifest_path = hand_dir / "render_manifest.json"
    release_memory()

    # 6. Tahap 8: Audit Kontrol Kualitas (QC)
    logger.info("Eksekusi Tahap 8: Audit Kontrol Kualitas Pasca-Render")
    qc_dir = session_workspace / "stage8_qc"
    qc_report = run_post_render_qc(
        rendered_video_path=rendered_video_path,
        output_dir=qc_dir,
        manifest_json_path=render_manifest_path
    )
    release_memory()

    total_duration_sec = round(time.time() - start_total_time, 2)
    logger.info("==========================================")
    logger.info("RINGKASAN EKSEKUSI PIPELINE KILAT")
    logger.info("==========================================")
    logger.info(f"Status Kualitas Akhir: {qc_report.get('status')}")
    logger.info(f"Total Waktu Proses: {total_duration_sec} detik")
    logger.info(f"Video Hasil Render: {rendered_video_path.resolve()}")
    logger.info(f"Laporan Audit QC: {(qc_dir / 'qc_report.json').resolve()}")

    if qc_report.get("status") == "FAIL":
        logger.warning("Catatan: Terdeteksi anomali pada video akhir. Periksa rincian pada qc_report.json.")
    else:
        logger.success("Video berhasil diproses secara otomatis dan memenuhi standar siaran.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Eksekusi pipeline terhenti akibat kesalahan kritis: {e}")
        sys.exit(1)