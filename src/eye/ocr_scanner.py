import ctypes
import gc
from pathlib import Path
from typing import List, Dict, Any
import cv2
from loguru import logger


def release_memory():
    """Memaksa Python dan kernel Linux membersihkan alokasi memori RAM."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def scan_video_ocr(
    video_path: Path,
    scene_list: List[Dict[str, Any]],
    enabled: bool = True,
    sample_keyframe_only: bool = True
) -> Dict[str, Any]:
    """
    Memindai keberadaan teks layar atau watermark pada video menggunakan PaddleOCR.
    Jika parameter enabled bernilai False, proses OCR dilewati secara instan tanpa
    memuat model ke memori RAM.
    
    Args:
        video_path: Path berkas video CFR masukan.
        scene_list: Daftar rentang waktu adegan hasil scene_detector.
        enabled: Saklar ON/OFF pemindaian OCR.
        sample_keyframe_only: Jika True, hanya memindai keyframe tengah adegan untuk efisiensi CPU.
        
    Returns:
        Dictionary data teks terdeteksi beserta koordinat batas poligonnya per adegan.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Berkas video tidak ditemukan: {video_path}")

    # Logika saklar OFF (Bypass instan)
    if not enabled:
        logger.info(f"Pemindaian OCR dimatikan (OFF). Melewati analisis teks layar untuk: {video_path.name}")
        ocr_bypass = {}
        for sc in scene_list:
            ocr_bypass[sc["scene_id"]] = []
        return ocr_bypass

    logger.info(f"Menjalankan PaddleOCR pada keyframe: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ocr_engine = None
    ocr_results = {}

    try:
        from paddleocr import PaddleOCR

        # Inisialisasi PaddleOCR untuk karakter alfabet Latin/Indonesia pada CPU
        ocr_engine = PaddleOCR(use_angle_cls=False, lang="latin")

        for sc in scene_list:
            sc_id = sc["scene_id"]
            keyframes = sc.get("keyframes", [sc["start_time"]])
            # Gunakan keyframe tengah adegan sebagai sampel representatif
            sample_time = keyframes[1] if len(keyframes) > 1 else keyframes[0]
            target_frame = int(sample_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                ocr_results[sc_id] = []
                continue

            h, w, _ = frame.shape
            # Downscale proporsional jika resolusi terlalu besar untuk mempercepat OCR
            scale = 1.0
            if w > 1280:
                scale = 1280.0 / w
                frame_resized = cv2.resize(frame, (1280, int(h * scale)))
            else:
                frame_resized = frame

            # Inferensi PaddleOCR
            result = ocr_engine.ocr(frame_resized, cls=False)
            scene_text_elements = []

            if result and result[0]:
                for line in result[0]:
                    box = line[0]  # Poligon koordinat [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = float(text_info[1])

                    if confidence >= 0.60:
                        # Normalisasi koordinat poligon ke skala 0.0 - 1.0
                        norm_polygon = []
                        for pt in box:
                            norm_x = round((pt[0] / scale) / w, 4)
                            norm_y = round((pt[1] / scale) / h, 4)
                            norm_polygon.append([norm_x, norm_y])

                        # Periksa apakah teks berada di area watermark sudut layar
                        avg_y = sum(p[1] for p in norm_polygon) / 4.0
                        is_watermark = avg_y < 0.15 or avg_y > 0.85

                        scene_text_elements.append({
                            "text": text,
                            "confidence": round(confidence, 3),
                            "polygon": norm_polygon,
                            "timestamp": round(sample_time, 3),
                            "is_watermark": is_watermark
                        })

            ocr_results[sc_id] = scene_text_elements

        logger.success(f"Pemindaian OCR selesai pada seluruh adegan.")
        return ocr_results

    except Exception as e:
        logger.warning(f"PaddleOCR mengalami kendala teknis ({e}). Menggunakan hasil kosong agar alur tidak mandek.")
        fallback_ocr = {}
        for sc in scene_list:
            fallback_ocr[sc["scene_id"]] = []
        return fallback_ocr
    finally:
        cap.release()
        del ocr_engine
        release_memory()