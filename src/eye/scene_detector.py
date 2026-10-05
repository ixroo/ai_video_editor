import ctypes
import gc
from pathlib import Path
from typing import List, Dict, Any
import cv2
from loguru import logger
from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector


def release_memory():
    """Memaksa Python dan kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def detect_scenes(
    video_path: Path,
    enabled: bool = True,
    threshold: float = 27.0,
    min_scene_len_sec: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Mendeteksi batas pergantian adegan fisik (cut) pada video menggunakan PySceneDetect.
    Jika parameter enabled bernilai False, modul mem-bypass pemindaian dan
    menetapkan seluruh video sebagai satu adegan tunggal.
    
    Args:
        video_path: Path berkas video CFR masukan.
        enabled: Saklar ON/OFF deteksi adegan.
        threshold: Ambang sensitivitas pendeteksian adegan adaptif.
        min_scene_len_sec: Durasi minimal adegan dalam satuan detik.
        
    Returns:
        Daftar dictionary metadata adegan beserta stempel waktu dan keyframe.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Berkas video tidak ditemukan: {video_path}")

    # Cek durasi dan framerate native berkas
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    # Logika saklar OFF (Bypass)
    if not enabled:
        logger.info(f"Deteksi adegan dimatikan (OFF). Menetapkan 1 adegan utuh untuk: {video_path.name}")
        return [
            {
                "scene_id": "scene_000",
                "start_time": 0.0,
                "end_time": round(duration_sec, 3),
                "start_frame": 0,
                "end_frame": int(frame_count),
                "duration": round(duration_sec, 3),
                "cut_type": "bypass",
                "keyframes": [
                    round(0.0, 3),
                    round(duration_sec / 2.0, 3),
                    round(max(0.0, duration_sec - 0.1), 3)
                ]
            }
        ]

    logger.info(f"Menjalankan PySceneDetect (AdaptiveDetector) pada: {video_path.name}")

    scene_manager = None
    video = None
    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        
        # Menggunakan AdaptiveDetector untuk akurasi tinggi pada variasi pencahayaan
        min_scene_frames = int(min_scene_len_sec * fps)
        scene_manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_frames
            )
        )

        scene_manager.detect_scenes(video=video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        results = []
        if not scene_list:
            logger.warning("Tidak ditemukan potongan adegan fisik. Menggunakan satu adegan penuh.")
            results.append({
                "scene_id": "scene_000",
                "start_time": 0.0,
                "end_time": round(duration_sec, 3),
                "start_frame": 0,
                "end_frame": int(frame_count),
                "duration": round(duration_sec, 3),
                "cut_type": "single_shot",
                "keyframes": [
                    round(0.0, 3),
                    round(duration_sec / 2.0, 3),
                    round(max(0.0, duration_sec - 0.1), 3)
                ]
            })
        else:
            for idx, (start_timecode, end_timecode) in enumerate(scene_list):
                s_sec = start_timecode.get_seconds()
                e_sec = end_timecode.get_seconds()
                s_frame = start_timecode.get_frames()
                e_frame = end_timecode.get_frames()
                dur = e_sec - s_sec
                mid_sec = s_sec + (dur / 2.0)

                results.append({
                    "scene_id": f"scene_{idx:03d}",
                    "start_time": round(s_sec, 3),
                    "end_time": round(e_sec, 3),
                    "start_frame": int(s_frame),
                    "end_frame": int(e_frame),
                    "duration": round(dur, 3),
                    "cut_type": "hard_cut",
                    "keyframes": [
                        round(s_sec, 3),
                        round(mid_sec, 3),
                        round(max(s_sec, e_sec - 0.05), 3)
                    ]
                })

        logger.success(f"Deteksi adegan selesai: Berhasil mengidentifikasi {len(results)} adegan.")
        return results

    except Exception as e:
        logger.error(f"Gagal saat mendeteksi adegan video: {e}")
        # Fallback darurat jika pustaka mengalami galat
        return [
            {
                "scene_id": "scene_000",
                "start_time": 0.0,
                "end_time": round(duration_sec, 3),
                "start_frame": 0,
                "end_frame": int(frame_count),
                "duration": round(duration_sec, 3),
                "cut_type": "fallback_single",
                "keyframes": [0.0, round(duration_sec / 2.0, 3), round(duration_sec, 3)]
            }
        ]
    finally:
        del scene_manager
        del video
        release_memory()