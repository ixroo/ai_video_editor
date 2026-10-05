import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from .scene_detector import detect_scenes
from .face_tracker import track_faces_and_reframe
from .ocr_scanner import scan_video_ocr


def run_eye_pipeline(
    video_path: Path,
    output_dir: Path,
    scene_detect_enabled: bool = True,
    face_track_enabled: bool = True,
    ocr_enabled: bool = False,
    target_start: Optional[float] = None,
    target_end: Optional[float] = None
) -> Dict[str, Any]:
    """
    Menjalankan alur Persepsi Visual (Tahap 3) secara selektif:
    Jika target_start & target_end diisi, sistem memfokuskan deteksi hanya pada
    rentang klip terpilih (hemat waktu dari 4 menit menjadi beberapa detik).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memulai eksekusi Tahap 3 (Ekstraksi Visual Selektif / Modul Mata)")

    # 1. Penentuan adegan: Lewati PySceneDetect jika rentang klip sudah dikunci oleh AI
    if target_start is not None and target_end is not None:
        logger.info(f"Fokus visual terkunci pada klip: {target_start:.2f}s - {target_end:.2f}s")
        scenes = [{
            "scene_id": "scene_01",
            "start_time": round(target_start, 3),
            "end_time": round(target_end, 3),
            "start_frame": int(target_start * 30),
            "end_frame": int(target_end * 30),
            "duration": round(target_end - target_start, 3),
            "cut_type": "cut",
            "keyframes": [round(target_start, 3)]
        }]
    else:
        scenes = detect_scenes(video_path, enabled=scene_detect_enabled)

    # 2. Pelacakan wajah dan pembingkaian 9:16
    reframe_data = track_faces_and_reframe(video_path, scenes, enabled=face_track_enabled)

    # 3. Pemindaian teks layar dan watermark
    ocr_data = scan_video_ocr(video_path, scenes, enabled=ocr_enabled)

    # 4. Penggabungan metadata per adegan
    consolidated_scenes = []
    for sc in scenes:
        sc_id = sc["scene_id"]
        sc_reframe = reframe_data.get(sc_id, {})
        sc_ocr = ocr_data.get(sc_id, [])

        consolidated_scenes.append({
            "scene_id": sc_id,
            "start_time": sc["start_time"],
            "end_time": sc["end_time"],
            "start_frame": sc["start_frame"],
            "end_frame": sc["end_frame"],
            "duration": sc["duration"],
            "cut_type": sc["cut_type"],
            "keyframes": sc["keyframes"],
            "reframe": {
                "mode": sc_reframe.get("mode", "static_center"),
                "face_detected": sc_reframe.get("face_detected", False),
                "optimal_center_x": sc_reframe.get("optimal_center_x", 0.50),
                "trajectories": sc_reframe.get("trajectories", [])
            },
            "ocr_elements": sc_ocr
        })

    visual_metadata = {
        "video_file": str(video_path.resolve()),
        "total_scenes": len(consolidated_scenes),
        "scenes": consolidated_scenes
    }

    output_json_path = output_dir / "visual_metadata.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(visual_metadata, f, indent=2, ensure_ascii=False)

    logger.success(f"Tahap 3 selesai. Berkas disimpan di {output_json_path.resolve()}")
    return visual_metadata