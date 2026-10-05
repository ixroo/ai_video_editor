import ctypes
import gc
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


def release_memory():
    """Memaksa Python dan kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def probe_file_technical_info(video_path: Path) -> Dict[str, Any]:
    """Membaca metadata teknis dasar dari berkas video menggunakan ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        meta = json.loads(res.stdout)
        v_stream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
        a_stream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), {})
        dur = float(meta.get("format", {}).get("duration", 0.0))
        return {
            "duration": dur,
            "width": int(v_stream.get("width", 0)),
            "height": int(v_stream.get("height", 0)),
            "r_frame_rate": v_stream.get("r_frame_rate", "30/1"),
            "has_audio": bool(a_stream)
        }
    except Exception as e:
        logger.error(f"Gagal membaca metadata teknis video: {e}")
        return {"duration": 0.0, "width": 0, "height": 0, "r_frame_rate": "0/1", "has_audio": False}


def detect_black_frames(
    video_path: Path,
    min_black_duration: float = 0.25,
    pixel_black_threshold: float = 0.10
) -> List[Dict[str, float]]:
    """Memindai seluruh frame untuk mendeteksi layar hitam akibat kegagalan rendering."""
    filter_str = f"blackdetect=d={min_black_duration}:pix_th={pixel_black_threshold}"
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-vf", filter_str,
        "-an", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    black_intervals = []
    
    # Ekstraksi output blackdetect dari log stderr FFmpeg
    pattern = re.compile(r"black_start:([0-9.]+)\s+black_end:([0-9.]+)\s+black_duration:([0-9.]+)")
    for match in pattern.finditer(res.stderr):
        start = float(match.group(1))
        end = float(match.group(2))
        dur = float(match.group(3))
        black_intervals.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(dur, 3)
        })
    return black_intervals


def detect_audio_metrics(video_path: Path) -> Dict[str, Any]:
    """
    Mengukur puncak volume dan rata-rata kekuatan audio.
    Kliping nyata hanya terpicu jika puncak suara melampaui 0.0 dB (intersample overshoot)
    atau jika puncak menyentuh 0.0 dB disertai volume rata-rata yang terlalu padat/panas (> -12 dB).
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", "volumedetect",
        "-vn", "-sn", "-dn", "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    max_vol = -99.0
    mean_vol = -99.0
    
    match_max = re.search(r"max_volume:\s*(-?[0-9.]+)\s*dB", res.stderr)
    if match_max:
        max_vol = float(match_max.group(1))
        
    match_mean = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", res.stderr)
    if match_mean:
        mean_vol = float(match_mean.group(1))
        
    # Kliping parah jika volume tembus di atas 0.0 dB atau audio over-compressed (> -12 dB)
    is_hard_clipping = max_vol > 0.05 or (max_vol >= 0.0 and mean_vol > -12.0)
    
    # Peringatan ringan jika puncak mendekati 0.0 dB tetapi rata-rata suara masih dinamis dan aman
    near_peak_warning = max_vol >= 0.0 and not is_hard_clipping
    
    return {
        "max_volume_db": max_vol,
        "mean_volume_db": mean_vol,
        "is_clipping": is_hard_clipping,
        "near_peak_warning": near_peak_warning
    }


def verify_rendered_video(
    rendered_video_path: Path,
    expected_duration: float,
    output_dir: Path,
    max_duration_tolerance: float = 0.35,
    max_allowed_black_duration: float = 0.30
) -> Dict[str, Any]:
    """
    Menjalankan audit kualitas pasca-render secara menyeluruh:
    1. Memeriksa keberadaan dan ukuran berkas video fisik
    2. Mendeteksi cacat visual frame hitam (black frame)
    3. Mengukur anomali audio kliping
    4. Membandingkan kesesuaian durasi rencana terhadap durasi nyata
    5. Menerbitkan berkas qc_report.json
    """
    video_path = Path(rendered_video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Memulai verifikasi QC pasca-render untuk: {video_path.name}")
    
    critical_issues: List[str] = []
    warnings_list: List[str] = []
    
    if not video_path.exists() or video_path.stat().st_size == 0:
        return {
            "status": "FAIL",
            "reason": "Berkas video hasil render tidak ditemukan atau berukuran 0 byte.",
            "issues": ["FILE_NOT_FOUND_OR_EMPTY"]
        }
        
    tech_info = probe_file_technical_info(video_path)
    actual_duration = tech_info["duration"]
    
    # 1. Audit Durasi
    duration_diff = abs(actual_duration - expected_duration)
    if duration_diff > max_duration_tolerance:
        critical_issues.append(
            f"Penyimpangan durasi melebihi batas toleransi: target {expected_duration:.2f}s, riil {actual_duration:.2f}s (selisih {duration_diff:.2f}s)"
        )
        
    # 2. Audit Layar Hitam
    black_intervals = detect_black_frames(video_path)
    critical_blacks = [b for b in black_intervals if b["duration"] >= max_allowed_black_duration]
    if critical_blacks:
        critical_issues.append(f"Ditemukan {len(critical_blacks)} segmen layar hitam dengan durasi bocor abnormal.")
        
    # 3. Audit Audio
    audio_info = {}
    if tech_info["has_audio"]:
        audio_info = detect_audio_metrics(video_path)
        if audio_info["is_clipping"]:
            critical_issues.append(
                f"Terdeteksi distorsi suara parah (hard clipping): max volume {audio_info['max_volume_db']} dB, mean {audio_info['mean_volume_db']} dB"
            )
        elif audio_info.get("near_peak_warning"):
            warnings_list.append(
                f"Puncak volume menyentuh {audio_info['max_volume_db']} dB, namun rata-rata suara sehat ({audio_info['mean_volume_db']} dB). Ditoleransi."
            )
    else:
        critical_issues.append("Berkas video tidak memuat jalur audio aktif.")
        
    overall_status = "PASS" if not critical_issues else "FAIL"
    
    qc_report = {
        "qc_version": "1.0.0",
        "status": overall_status,
        "video_path": str(video_path.resolve()),
        "file_size_bytes": video_path.stat().st_size,
        "technical_info": tech_info,
        "expected_duration": expected_duration,
        "actual_duration": actual_duration,
        "duration_deviation": round(duration_diff, 3),
        "black_frame_intervals": black_intervals,
        "audio_metrics": audio_info,
        "detected_issues": critical_issues,
        "warnings": warnings_list
    }
    
    report_path = output_dir / "qc_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(qc_report, f, indent=2, ensure_ascii=False)
        
    if overall_status == "PASS":
        logger.success(f"Audit QC berhasil: Video dinyatakan LULUS (PASS). Laporan: {report_path.name}")
        if warnings_list:
            for w in warnings_list:
                logger.info(f"Catatan QC: {w}")
    else:
        logger.warning(f"Audit QC menemukan anomali (FAIL). Detail masalah tercatat di {report_path.name}")
        
    release_memory()
    return qc_report