import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .post_verifier import verify_rendered_video


def run_post_render_qc(
    rendered_video_path: Path,
    output_dir: Path,
    manifest_json_path: Optional[Path] = None,
    expected_duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Menjalankan orkestrator lengkap Tahap 8 (Kontrol Kualitas Pasca-Render):
    1. Membaca target durasi dari manifes render atau argumen langsung
    2. Menjalankan audit fisik video (layar hitam, audio kliping, dan durasi)
    3. Menerbitkan berkas qc_report.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_duration = 0.0
    if expected_duration is not None:
        target_duration = float(expected_duration)
    elif manifest_json_path and Path(manifest_json_path).exists():
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            target_duration = float(manifest_data.get("total_duration", 0.0))
            
    logger.info("==========================================")
    logger.info(" MEMULAI AUDIT TAHAP 8 (POST-RENDER QC)")
    logger.info("==========================================")
    
    report = verify_rendered_video(
        rendered_video_path=rendered_video_path,
        expected_duration=target_duration,
        output_dir=output_dir
    )
    
    return report


__all__ = [
    "run_post_render_qc",
    "verify_rendered_video"
]