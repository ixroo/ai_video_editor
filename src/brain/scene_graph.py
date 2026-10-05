import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Membaca dan memvalidasi berkas JSON dari disk."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Berkas JSON tidak ditemukan: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_words_to_scenes(
    speech_segments: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Memetakan kalimat dan kata-kata hasil transkripsi ke adegan visual spesifik
    berdasarkan irisan stempel waktu.
    """
    enriched_scenes = []

    for sc in scenes:
        sc_start = sc["start_time"]
        sc_end = sc["end_time"]
        matched_sentences = []
        scene_words_count = 0

        for seg in speech_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            # Cek apakah segmen bicara beririsan dengan adegan ini
            if (seg_start <= sc_end) and (seg_end >= sc_start):
                # Filter kata-kata yang spesifik berada di rentang adegan
                in_scene_words = [
                    w for w in seg.get("words", [])
                    if sc_start <= w["start"] <= sc_end or sc_start <= w["end"] <= sc_end
                ]
                scene_words_count += len(in_scene_words)

                matched_sentences.append({
                    "segment_id": seg.get("id"),
                    "speaker_id": seg.get("speaker_id", "spk_0"),
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg.get("text", ""),
                    "words_in_scene": in_scene_words
                })

        enriched_sc = dict(sc)
        enriched_sc["speech"] = {
            "sentences": matched_sentences,
            "total_words": scene_words_count,
            "has_speech": len(matched_sentences) > 0
        }
        enriched_scenes.append(enriched_sc)

    return enriched_scenes


def generate_compact_llm_summary(scene_graph: Dict[str, Any]) -> str:
    """
    Menghasilkan ringkasan semantik terstruktur yang padat dan hemat token
    untuk dikirim ke Gemini API tanpa menghabiskan kuota TPM Free Tier.
    """
    lines = []
    lines.append(f"TOTAL_DURATION: {scene_graph['duration_seconds']}s")
    lines.append(f"TOTAL_SCENES: {len(scene_graph['scenes'])}")
    lines.append(f"TOTAL_SILENCE_INTERVALS: {len(scene_graph['silence_intervals'])}")
    lines.append("\n=== DAFTAR ADEGAN & TRANSKRIP ===")

    for sc in scene_graph["scenes"]:
        sc_id = sc["scene_id"]
        t_start = sc["start_time"]
        t_end = sc["end_time"]
        face_info = "Face:Ada" if sc["reframe"]["face_detected"] else "Face:Tidak"
        center_x = sc["reframe"]["optimal_center_x"]

        # Kumpulkan teks dialog pada adegan ini
        dialogues = []
        for s in sc.get("speech", {}).get("sentences", []):
            spk = s["speaker_id"]
            txt = s["text"]
            dialogues.append(f"[{spk}] {txt}")

        dialogue_str = " | ".join(dialogues) if dialogues else "(Hening / Tidak ada ucapan)"
        lines.append(f"[{sc_id}] ({t_start}s - {t_end}s) | {face_info} (cx={center_x}) | {dialogue_str}")

    return "\n".join(lines)


def build_unified_scene_graph(
    acoustic_data: Dict[str, Any],
    visual_data: Dict[str, Any],
    technical_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Menyatukan data akustik dan visual ke dalam skema Scene Graph kanonikal.
    """
    total_duration = acoustic_data.get("total_duration", 0.0)
    speech_segments = acoustic_data.get("speech_segments", [])
    silence_intervals = acoustic_data.get("silence_intervals", [])
    raw_scenes = visual_data.get("scenes", [])

    # 1. Korelasikan kata transkripsi dengan rentang adegan visual
    scenes_with_speech = map_words_to_scenes(speech_segments, raw_scenes)

    # 2. Susun dokumen Scene Graph terpadu
    scene_graph = {
        "schema_version": "1.0.0",
        "duration_seconds": total_duration,
        "technical_info": technical_info or {},
        "silence_intervals": silence_intervals,
        "acoustic_summary": {
            "language": acoustic_data.get("language", "id"),
            "language_probability": acoustic_data.get("language_probability", 1.0),
            "full_transcript": acoustic_data.get("full_text", "")
        },
        "scenes": scenes_with_speech
    }

    # 3. Buat ringkasan padat untuk prompt model perencana (Otak)
    compact_summary = generate_compact_llm_summary(scene_graph)
    scene_graph["llm_compact_summary"] = compact_summary

    return scene_graph


def run_scene_graph_pipeline(
    acoustic_json_path: Path,
    visual_json_path: Path,
    output_dir: Path,
    manifest_json_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Menjalankan alur lengkap Sinkronisasi Scene Graph (Tahap 4):
    1. Membaca metadata audio dan video dari disk
    2. Menyelaraskan stempel waktu ke satu basis kanonikal
    3. Memetakan kata ucapan dan pembicara ke kotak adegan visual
    4. Menyusun ringkasan hemat token untuk Gemini API
    5. Menyimpan hasil ke scene_graph.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memulai eksekusi Tahap 4 (Sinkronisasi Scene Graph / Modul Otak)")

    # Muat dokumen prasyarat
    acoustic_data = load_json_file(acoustic_json_path)
    visual_data = load_json_file(visual_json_path)

    tech_info = {}
    if manifest_json_path and Path(manifest_json_path).exists():
        manifest_data = load_json_file(manifest_json_path)
        tech_info = manifest_data.get("technical_info", {})

    # Bentuk Scene Graph kanonikal
    scene_graph = build_unified_scene_graph(acoustic_data, visual_data, tech_info)

    output_path = output_dir / "scene_graph.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_graph, f, indent=2, ensure_ascii=False)

    logger.success(f"Tahap 4 selesai. Dokumen kanonikal tersimpan di: {output_path.resolve()}")
    return scene_graph