import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .otio_compiler import run_otio_compiler_pipeline
from .subtitle_generator import run_subtitle_pipeline
from .ffmpeg_vaapi import run_ffmpeg_render_pipeline


def run_hand_pipeline(
    validated_plan_path: Path,
    scene_graph_path: Path,
    output_dir: Path,
    video_path: Optional[Path] = None,
    manifest_json_path: Optional[Path] = None,
    fps: int = 30,
    enable_subtitles: bool = True,
    use_hardware: bool = True,
    vaapi_device: str = "/dev/dri/renderD128",
    output_name: str = "final_render.mp4",
    words_per_line: int = 3,
    font_name: str = "DejaVu Sans",
    font_size: int = 72,
    margin_bottom: int = 460,
    uppercase: bool = False,
    keep_temp: bool = False
) -> Dict[str, Any]:
    """
    Menjalankan orkestrator lengkap kluster Tangan (Tahap 7):
    1. Tahap 7A: Kompilasi timeline OTIO dan manifes render presisi frame
    2. Tahap 7B: Pembuatan subtitle highlight kata aktif
    3. Tahap 7C: Render FFmpeg (VAAPI, fallback CPU) dan pembakaran subtitle

    Args:
        validated_plan_path: Path validated_plan.json hasil Tahap 6.
        scene_graph_path: Path scene_graph.json hasil Tahap 4.
        output_dir: Direktori kerja dan hasil render.
        video_path: Path video CFR hasil Tahap 1 (opsional jika manifest diberikan).
        manifest_json_path: Path ingest_manifest.json untuk membaca normalized_video_path.
        enable_subtitles: Saklar ON/OFF pembakaran subtitle.
        use_hardware: Saklar ON/OFF akselerasi VAAPI.

    Returns:
        Dictionary hasil render (path video akhir, durasi, jumlah job hardware/CPU).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if video_path is None:
        if manifest_json_path is None:
            raise ValueError("Isi salah satu: video_path atau manifest_json_path.")
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            video_path = Path(json.load(f)["normalized_video_path"])
    video_path = Path(video_path)

    logger.info("==========================================")
    logger.info(" MEMULAI ORKESTRASI MODUL TANGAN (RENDER)")
    logger.info("==========================================")

    # 1. Tahap 7A - Timeline OTIO dan manifes render
    render_manifest = run_otio_compiler_pipeline(
        validated_plan_path=validated_plan_path,
        video_path=video_path,
        output_dir=output_dir,
        scene_graph_path=scene_graph_path,
        fps=fps
    )

    # 2. Tahap 7B - Subtitle
    subtitle_path = None
    if enable_subtitles:
        subtitle_result = run_subtitle_pipeline(
            scene_graph_path=scene_graph_path,
            validated_plan_path=validated_plan_path,
            output_dir=output_dir,
            fps=fps,
            words_per_line=words_per_line,
            play_res_x=render_manifest["output_width"],
            play_res_y=render_manifest["output_height"],
            font_name=font_name,
            font_size=font_size,
            margin_bottom=margin_bottom,
            uppercase=uppercase
        )
        subtitle_path = Path(subtitle_result["subtitle_path"])
    else:
        logger.info("Pembakaran subtitle dimatikan (OFF).")

    # 3. Tahap 7C - Render fisik
    render_result = run_ffmpeg_render_pipeline(
        manifest=render_manifest,
        output_dir=output_dir,
        subtitle_path=subtitle_path,
        output_name=output_name,
        use_hardware=use_hardware,
        vaapi_device=vaapi_device,
        keep_temp=keep_temp
    )

    logger.success("Seluruh alur modul Tangan (Tahap 7) selesai dieksekusi.")
    return render_result


__all__ = [
    "run_hand_pipeline",
    "run_otio_compiler_pipeline",
    "run_subtitle_pipeline",
    "run_ffmpeg_render_pipeline",
]