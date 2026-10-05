import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .scene_graph import run_scene_graph_pipeline
from .gemini_planner import run_gemini_planner_pipeline
from .contract_gatekeeper import run_contract_gatekeeper_pipeline


def run_brain_pipeline(
    acoustic_json_path: Path,
    visual_json_path: Optional[Path] = None,
    output_dir: Path = Path("storage/workspace/stage4_brain"),
    manifest_json_path: Optional[Path] = None,
    model_name: str = "gemini-2.5-flash",
    fallback_model: str = "gemini-2.5-flash",
    target_max_duration: float = 60.0
) -> Dict[str, Any]:
    """
    Menjalankan orkestrator kluster Otak dan Gerbang Kontrak.
    Mendukung mode Audio-First (visual_json_path=None) sehingga Gemini bisa
    merencanakan narasi langsung dari transkrip tanpa menunggu pelacakan visual.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    acoustic_json_path = Path(acoustic_json_path)

    logger.info("==========================================")
    logger.info("   MEMULAI ORKESTRASI MODUL OTAK & KONTRAK")
    logger.info("==========================================")

    with open(acoustic_json_path, "r", encoding="utf-8") as f:
        acoustic_data = json.load(f)
    total_dur = float(acoustic_data.get("total_duration", 60.0))

    # Buat visual metadata sementara jika berjalan dalam mode Audio-First
    if visual_json_path is None or not Path(visual_json_path).exists():
        logger.info("Mode Audio-First aktif - Membangun Scene Graph langsung dari transkrip ucapan.")
        temp_visual = output_dir / "preliminary_visual_metadata.json"
        with open(temp_visual, "w", encoding="utf-8") as f:
            json.dump({
                "video_file": str(acoustic_json_path),
                "total_scenes": 1,
                "scenes": [{
                    "scene_id": "scene_001",
                    "start_time": 0.0,
                    "end_time": round(total_dur, 3),
                    "start_frame": 0,
                    "end_frame": int(total_dur * 30),
                    "duration": round(total_dur, 3),
                    "cut_type": "cut",
                    "keyframes": [0.0],
                    "speech": {
                        "has_speech": True,
                        "speech_ratio": float(acoustic_data.get("speech_ratio", 0.9))
                    },
                    "reframe": {
                        "mode": "static_center",
                        "face_detected": False,
                        "optimal_center_x": 0.50,
                        "trajectories": []
                    },
                    "ocr_elements": []
                }]
            }, f, indent=2)
        visual_json_path = temp_visual

    # 1. Eksekusi Tahap 4 - Sinkronisasi Scene Graph
    scene_graph = run_scene_graph_pipeline(
        acoustic_json_path=acoustic_json_path,
        visual_json_path=visual_json_path,
        output_dir=output_dir,
        manifest_json_path=manifest_json_path
    )
    scene_graph_path = output_dir / "scene_graph.json"

    # Pastikan data transkripsi selalu tersedia di berkas scene_graph.json untuk Gemini
    if "transcription" not in scene_graph or not scene_graph["transcription"].get("segments"):
        if "transcription" in acoustic_data:
            scene_graph["transcription"] = acoustic_data["transcription"]
            with open(scene_graph_path, "w", encoding="utf-8") as f:
                json.dump(scene_graph, f, indent=2, ensure_ascii=False)

    # 2. Eksekusi Tahap 5 - Perencanaan Narasi AI
    draft_plan = run_gemini_planner_pipeline(
        scene_graph_path=scene_graph_path,
        output_dir=output_dir,
        model_name=model_name,
        fallback_model=fallback_model,
        target_max_duration=target_max_duration
    )
    draft_plan_path = output_dir / "draft_plan.json"

    # 3. Eksekusi Tahap 6 - Gerbang Kontrak Deterministik
    validated_contract = run_contract_gatekeeper_pipeline(
        draft_plan_path=draft_plan_path,
        scene_graph_path=scene_graph_path,
        output_dir=output_dir,
        max_duration=target_max_duration
    )

    logger.success("Seluruh alur modul Otak selesai dieksekusi dengan sukses.")
    return validated_contract