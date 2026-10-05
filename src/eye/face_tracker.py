import ctypes
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import mediapipe as mp
from loguru import logger


def release_memory():
    """Memaksa Python dan kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def smooth_trajectory(values: List[float], window_size: int = 5) -> List[float]:
    """Menerapkan perataan moving average agar pergerakan crop kamera tidak patah-patah."""
    if not values:
        return []
    smoothed = []
    half = window_size // 2
    for i in range(len(values)):
        start_idx = max(0, i - half)
        end_idx = min(len(values), i + half + 1)
        window = values[start_idx:end_idx]
        smoothed.append(round(sum(window) / len(window), 4))
    return smoothed


def track_faces_and_reframe(
    video_path: Path,
    scene_list: List[Dict[str, Any]],
    enabled: bool = True,
    sample_interval_sec: float = 0.5,
    min_detection_confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Melacak posisi wajah manusia per adegan menggunakan MediaPipe Face Detection.
    Mendukung seleksi wajah paling dominan saat multi-wajah dan memiliki
    saklar ON/OFF untuk bypass instan ke mode crop tengah (center crop).
    
    Args:
        video_path: Path berkas video CFR.
        scene_list: Daftar rentang waktu adegan hasil scene_detector.
        enabled: Saklar ON/OFF pelacak wajah.
        sample_interval_sec: Interval pengambilan sampel frame (detik) untuk efisiensi CPU.
        min_detection_confidence: Ambang keyakinan deteksi model.
        
    Returns:
        Dictionary berisi riwayat trajektori koordinat crop optimal per adegan.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Berkas video tidak ditemukan: {video_path}")

    # Logika saklar OFF (Bypass tanpa memuat model AI ke memori)
    if not enabled:
        logger.info(f"Pelacakan wajah dimatikan (OFF). Mengunci crop tengah statis (center_x = 0.50).")
        reframe_data = {}
        for sc in scene_list:
            sc_id = sc["scene_id"]
            reframe_data[sc_id] = {
                "mode": "static_center",
                "face_detected": False,
                "optimal_center_x": 0.50,
                "trajectories": [
                    {"timestamp": sc["start_time"], "center_x": 0.50},
                    {"timestamp": sc["end_time"], "center_x": 0.50}
                ]
            }
        return reframe_data

    logger.info(f"Menjalankan MediaPipe Face Detection pada: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * sample_interval_sec))

    mp_face_detection = mp.solutions.face_detection
    face_detector = None
    reframe_data = {}

    try:
        # Inisialisasi model MediaPipe pada CPU
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 untuk jangkauan deteksi kamera penuh
            min_detection_confidence=min_detection_confidence
        )

        for sc in scene_list:
            sc_id = sc["scene_id"]
            start_frame = sc["start_frame"]
            end_frame = sc["end_frame"]

            raw_centers = []
            timestamps = []
            detected_any = False

            current_frame_idx = start_frame
            while current_frame_idx <= end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                t_sec = current_frame_idx / fps
                h, w, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Jalankan inferensi deteksi wajah
                detection_result = face_detector.process(frame_rgb)

                if detection_result.detections:
                    detected_any = True
                    # Seleksi wajah dominan (pilih wajah dengan area bounding box terbesar)
                    largest_face = None
                    max_area = -1.0

                    for det in detection_result.detections:
                        bbox = det.location_data.relative_bounding_box
                        area = bbox.width * bbox.height
                        if area > max_area:
                            max_area = area
                            largest_face = bbox

                    if largest_face:
                        center_x = largest_face.xmin + (largest_face.width / 2.0)
                        # Batasi titik potong agar tidak melewati batas fisik kanvas
                        clamped_center = max(0.28, min(0.72, center_x))
                        raw_centers.append(clamped_center)
                    else:
                        # Fallback jika bbox tidak valid
                        raw_centers.append(raw_centers[-1] if raw_centers else 0.50)
                else:
                    # Skenario wajah tidak ditemukan di frame ini: tahan di posisi sebelumnya
                    raw_centers.append(raw_centers[-1] if raw_centers else 0.50)

                timestamps.append(round(t_sec, 3))
                current_frame_idx += frame_interval

            # Terapkan temporal smoothing pada pergerakan kamera adegan ini
            smoothed_centers = smooth_trajectory(raw_centers, window_size=5)

            trajectories = []
            for t, cx in zip(timestamps, smoothed_centers):
                trajectories.append({"timestamp": t, "center_x": cx})

            avg_center = (
                round(sum(smoothed_centers) / len(smoothed_centers), 4)
                if smoothed_centers else 0.50
            )

            reframe_data[sc_id] = {
                "mode": "auto_reframe" if detected_any else "fallback_center",
                "face_detected": detected_any,
                "optimal_center_x": avg_center,
                "trajectories": trajectories
            }

        logger.success(f"Pelacakan visual selesai: {len(reframe_data)} adegan terpetakan.")
        return reframe_data

    except Exception as e:
        logger.error(f"Kegagalan saat menjalankan pelacakan wajah: {e}")
        # Fallback darurat
        fallback_data = {}
        for sc in scene_list:
            fallback_data[sc["scene_id"]] = {
                "mode": "error_center_fallback",
                "face_detected": False,
                "optimal_center_x": 0.50,
                "trajectories": [{"timestamp": sc["start_time"], "center_x": 0.50}]
            }
        return fallback_data
    finally:
        cap.release()
        del face_detector
        release_memory()