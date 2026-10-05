import os
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


# Daftar urutan model awan paket gratis dari prioritas tertinggi ke terendah
FALLBACK_MODEL_CHAIN = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash"
]

# Kata sambung/penghubung yang dilarang keras berada di akhir video
DANGLING_CONJUNCTIONS = {
    "dan", "atau", "tapi", "tetapi", "karena", "sebab", "bahwa", "yang",
    "sehingga", "maka", "terus", "lalu", "namun", "jika", "kalau", "nah",
    "kan", "jadi", "soalnya", "secara", "buat", "untuk", "seperti", "kayak"
}


def load_env_file(env_path: Path = Path(".env")) -> None:
    """Membaca berkas .env secara manual tanpa dependensi eksternal."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')
    except Exception as e:
        logger.warning(f"Gagal membaca berkas .env - {e}")


def generate_local_rule_based_plan(scene_graph: Dict[str, Any], target_max_duration: float = 60.0) -> Dict[str, Any]:
    """
    Rencana cadangan lokal berbasis aturan jika seluruh API Gemini awan mengalami kendala.
    Membuang jeda hening dan mengambil segmen aktif hingga durasi target tercapai.
    """
    logger.info("Menjalankan perencana editorial cadangan berbasis aturan lokal (Offline Fallback)...")
    scenes = scene_graph.get("scenes", [])
    silence_intervals = scene_graph.get("silence_intervals", [])
    
    selected_clips = []
    accumulated_duration = 0.0

    for sc in scenes:
        sc_start = sc["start_time"]
        sc_end = sc["end_time"]

        clip_start = sc_start
        clip_end = sc_end

        for sil in silence_intervals:
            if abs(sil["start"] - clip_start) < 0.2:
                clip_start = min(clip_end, sil["end"])
            if abs(sil["end"] - clip_end) < 0.2:
                clip_end = max(clip_start, sil["start"])

        clip_duration = clip_end - clip_start
        if clip_duration < 0.8:
            continue

        if accumulated_duration + clip_duration > target_max_duration:
            allowed_duration = target_max_duration - accumulated_duration
            if allowed_duration > 1.5:
                clip_end = clip_start + allowed_duration
                clip_duration = allowed_duration
            else:
                break

        reframe_cfg = sc.get("reframe", {})
        optimal_cx = reframe_cfg.get("optimal_center_x", 0.50)

        selected_clips.append({
            "clip_id": f"clip_{len(selected_clips) + 1:02d}",
            "scene_id": sc["scene_id"],
            "source_start": round(clip_start, 3),
            "source_end": round(clip_end, 3),
            "duration": round(clip_duration, 3),
            "optimal_center_x": optimal_cx,
            "reframe_mode": reframe_cfg.get("mode", "static_center"),
            "narrative_purpose": "Dialog aktif segmen pembahasan",
            "cleaned_transcript": ""
        })

        accumulated_duration += clip_duration
        if accumulated_duration >= target_max_duration:
            break

    if not selected_clips and scenes:
        first_sc = scenes[0]
        end_time = min(first_sc["start_time"] + target_max_duration, first_sc["end_time"])
        selected_clips.append({
            "clip_id": "clip_01",
            "scene_id": first_sc["scene_id"],
            "source_start": round(first_sc["start_time"], 3),
            "source_end": round(end_time, 3),
            "duration": round(end_time - first_sc["start_time"], 3),
            "optimal_center_x": 0.50,
            "reframe_mode": "static_center",
            "narrative_purpose": "Klip awal pembuka video",
            "cleaned_transcript": ""
        })
        accumulated_duration = end_time - first_sc["start_time"]

    hook_segment = {}
    if selected_clips:
        first_clip = selected_clips[0]
        hook_duration = min(4.0, first_clip["duration"])
        hook_segment = {
            "source_start": first_clip["source_start"],
            "source_end": round(first_clip["source_start"] + hook_duration, 3),
            "headline": "Sorotan Percakapan Utama",
            "reason": "Segmen pembuka dengan energi bicara aktif"
        }

    return {
        "status": "success",
        "plan_source": "local_rule_based_fallback",
        "model_used": "deterministic_rules",
        "target_aspect_ratio": "9:16",
        "total_planned_duration": round(accumulated_duration, 3),
        "hook_segment": hook_segment,
        "selected_clips": selected_clips,
        "cleaned_transcript": "",
        "editorial_reasoning": "Dihasilkan secara deterministik tanpa LLM awan."
    }


SYSTEM_PROMPT = """
Anda adalah asisten sutradara dan editor video profesional spesialis video pendek vertikal (Shorts/Reels/TikTok).
Tugas utama Anda adalah mengurasi SATU rentang klip narasi terbaik yang berstruktur UTUH, MEMIKAT, dan TIDAK BOLEH MENGGANTUNG.

PANDUAN KURASI WAJIB:
1. DILARANG KERAS MEMILIH INTRO BASA-BASI: Jangan mengambil segmen di bawah detik ke-60 jika isinya hanya salam pembuka, obrolan santai tanpa isi, sponsor, atau perkenalan diri. Cari momen di mana inti argumen atau pembahasan berbobot sedang memuncak.
2. STRUKTUR TIGA BABAK NYATA:
   - Hook (3-5 detik pembuka): Diawali pernyataan mengejutkan, opini kontroversial, atau pertanyaan menohok yang langsung memicu rasa ingin tahu penonton.
   - Isi (alur tengah): Penjelasan logis, data, atau analogi konkret yang membuktikan atau menjawab pernyataan hook tersebut.
   - Penutup (3-5 detik akhir): Kesimpulan tuntas, punchline cerdas, atau kalimat penutup mutlak.
3. ANTI-MENGGANTUNG: Kalimat terakhir pada segmen penutup WAJIB tuntas secara struktur tata bahasa. DILARANG BERHENTI di tengah anak kalimat atau kata penghubung (seperti: 'tapi', 'karena', 'dan', 'sehingga', 'terus', 'kalau').
4. Durasi total klip harus berkisar antara 22.0 hingga 55.0 detik.

ATURAN KELUARAN:
- Respon WAJIB JSON murni tanpa markdown pembuka/penutup.
- Cantumkan kutipan kalimat persis untuk hook, argumen isi, dan punchline penutup.
"""


def call_gemini_api(
    prompt_text: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Memanggil Google GenAI SDK untuk mendapatkan rekomendasi garis waktu editorial."""
    warnings.filterwarnings("ignore", message=".*automatic function calling.*")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    if not response.text:
        raise ValueError("Respon API Gemini kosong.")

    return json.loads(response.text)


def extract_segments_from_scene_graph(scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mengekstrak seluruh segmen transkripsi ucapan dari metadata scene graph."""
    if "transcription" in scene_graph and isinstance(scene_graph["transcription"], dict):
        raw_segs = scene_graph["transcription"].get("segments", [])
        if raw_segs:
            return raw_segs

    for key in ["acoustic_metadata", "acoustic"]:
        if key in scene_graph and isinstance(scene_graph[key], dict):
            raw_segs = scene_graph[key].get("transcription", {}).get("segments", [])
            if raw_segs:
                return raw_segs

    segments = []
    scenes = scene_graph.get("scenes", [])
    for sc in scenes:
        speech = sc.get("speech", {})
        sc_segs = speech.get("segments", [])
        if sc_segs:
            segments.extend(sc_segs)
        elif "sentences" in speech and speech["sentences"]:
            for sent in speech["sentences"]:
                segments.append({
                    "id": sent.get("sentence_id") or sent.get("id"),
                    "start": sent.get("start_time") or sent.get("start", 0.0),
                    "end": sent.get("end_time") or sent.get("end", 0.0),
                    "text": sent.get("text", "")
                })
            
    return segments


def ensure_clean_ending(
    all_segments: List[Dict[str, Any]],
    start_seg: Dict[str, Any],
    end_seg: Dict[str, Any],
    max_duration: float
) -> Dict[str, Any]:
    """
    Memeriksa kata terakhir pada segmen penutup. Jika berujung pada kata sambung
    (menggantung), fungsi ini secara deterministik memajukan titik potong ke
    segmen berikutnya yang berakhiran tanda baca atau kalimat tuntas.
    """
    curr_end_idx = all_segments.index(end_seg)
    start_time = float(start_seg["start"])

    while curr_end_idx < len(all_segments) - 1:
        current_text = all_segments[curr_end_idx]["text"].strip()
        words = re.findall(r"\b\w+\b", current_text.lower())
        last_word = words[-1] if words else ""

        # Jika kata terakhir adalah kata sambung yang menggantung
        if last_word in DANGLING_CONJUNCTIONS:
            next_seg = all_segments[curr_end_idx + 1]
            candidate_duration = float(next_seg["end"]) - start_time
            if candidate_duration <= (max_duration + 2.0):
                logger.info(
                    f"Terdeteksi kata menggantung ('{last_word}') pada akhir segmen {all_segments[curr_end_idx].get('id')}. "
                    f"Memajukan titik potong ke segmen {next_seg.get('id')} agar kalimat tuntas."
                )
                curr_end_idx += 1
                continue
        break

    return all_segments[curr_end_idx]


def plan_editorial_with_gemini(
    scene_graph: Dict[str, Any],
    model_name: str = "gemini-3.7-flash",
    fallback_model: str = "gemini-3.6-flash",
    target_max_duration: float = 60.0
) -> Dict[str, Any]:
    """
    Menyusun draf editorial dengan kurasi 3 babak nyata dan arsitektur Cascading Fallback.
    Dilengkapi filter anti-basa-basi dan penutup deterministik bebas kalimat menggantung.
    """
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "masukkan_api_key_lo_di_sini":
        logger.warning("GEMINI_API_KEY belum dikonfigurasi di berkas .env. Mengalihkan ke perencana lokal.")
        return generate_local_rule_based_plan(scene_graph, target_max_duration)

    all_segments = extract_segments_from_scene_graph(scene_graph)

    if not all_segments:
        logger.warning("Data segmen ucapan kosong pada Scene Graph. Menggunakan perencana aturan lokal.")
        return generate_local_rule_based_plan(scene_graph, target_max_duration)

    formatted_transcript = []
    for s in all_segments:
        seg_id = s.get("id")
        start = s.get("start", 0.0)
        end = s.get("end", 0.0)
        text = s.get("text", "").strip()
        if text:
            formatted_transcript.append({
                "segment_id": seg_id,
                "start": start,
                "end": end,
                "text": text
            })

    user_query = f"""
Daftar seluruh segmen transkripsi ucapan lengkap:
{json.dumps(formatted_transcript, ensure_ascii=False, indent=1)}

INSTRUKSI KURATOR:
1. Abaikan basa-basi pembuka video di awal durasi (intro, sapaan halo, perkenalan). Pilih segmen pembahasan inti di mana argumen menarik atau cerita utama sedang dibicarakan.
2. Cari SATU alur narasi berdurasi antara 22.0 hingga {target_max_duration} detik dengan struktur:
   - Hook: Kalimat awal yang membuat penonton tidak mau scroll
   - Isi: Penjelasan inti yang menjawab topik hook
   - Penutup: Kalimat simpulan yang benar-benar selesai (tidak menggantung di kata sambung)
3. Tentukan `start_segment_id` dan `end_segment_id` dari daftar di atas.
4. Tuliskan teks segmen terpilih pada `cleaned_transcript` dengan ejaan rapi.

Format respon WAJIB JSON murni:
{{
  "start_segment_id": int,
  "end_segment_id": int,
  "hook_quote": "Kutipan kalimat pembuka pemikat",
  "body_quote": "Kutipan inti penjelasan pembahasan",
  "punchline_conclusion_quote": "Kutipan kalimat penutup yang tuntas",
  "headline_hook": "Judul headline pemikat 3 detik awal",
  "cleaned_transcript": "Seluruh teks percakapan klip yang rapi",
  "editorial_reasoning": "Alasan mengapa alur ini utuh dan tidak menggantung"
}}
"""

    model_queue = [model_name]
    if fallback_model and fallback_model not in model_queue:
        model_queue.append(fallback_model)
    for m in FALLBACK_MODEL_CHAIN:
        if m not in model_queue:
            model_queue.append(m)

    gemini_raw_plan = None
    successful_model = None

    for target_m in model_queue:
        try:
            logger.info(f"Mengirim permintaan perencana narasi ke model: {target_m}")
            gemini_raw_plan = call_gemini_api(user_query, api_key, target_m)
            successful_model = target_m
            logger.success(f"Perencanaan narasi berhasil disusun oleh model: {target_m}")
            break
        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "UNAVAILABLE" in err_msg or "high demand" in err_msg:
                logger.warning(f"Model {target_m} sedang kebanjiran antrean (503 High Demand). Langsung beralih ke model cadangan...")
            elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                logger.warning(f"Model {target_m} menyentuh batas frekuensi (429 Rate Limit). Langsung beralih ke model cadangan...")
            else:
                logger.warning(f"Model {target_m} mengalami kendala: {err_msg[:200]}. Beralih ke model cadangan...")

    if not gemini_raw_plan:
        logger.warning("Seluruh model awan pada rantai cadangan gagal. Mengaktifkan mode pemotong deterministik lokal.")
        return generate_local_rule_based_plan(scene_graph, target_max_duration)

    try:
        start_id = gemini_raw_plan.get("start_segment_id")
        end_id = gemini_raw_plan.get("end_segment_id")

        matching_starts = [s for s in all_segments if s.get("id") == start_id]
        matching_ends = [s for s in all_segments if s.get("id") == end_id]

        if not matching_starts or not matching_ends:
            logger.warning("ID segmen dari respons Gemini tidak cocok. Menggunakan fallback lokal.")
            return generate_local_rule_based_plan(scene_graph, target_max_duration)

        start_seg = matching_starts[0]
        chosen_end_seg = matching_ends[-1]

        # Validasi anti-menggantung secara deterministik
        end_seg = ensure_clean_ending(all_segments, start_seg, chosen_end_seg, target_max_duration)

        source_start = float(start_seg["start"])
        raw_source_end = float(end_seg["end"])

        # Tambahkan ruang napas buffer udara 1.0 detik di akhir klip agar tidak close kasar
        end_idx = all_segments.index(end_seg)
        tail_buffer = 1.0
        if end_idx + 1 < len(all_segments):
            next_seg_start = float(all_segments[end_idx + 1]["start"])
            available_gap = max(0.0, next_seg_start - raw_source_end)
            source_end = raw_source_end + min(tail_buffer, max(0.4, available_gap - 0.1))
        else:
            source_end = raw_source_end + tail_buffer

        if source_end <= source_start or (source_end - source_start) > (target_max_duration + 5.0):
            source_end = min(source_start + target_max_duration, float(all_segments[-1]["end"]))

        clip_duration = round(source_end - source_start, 3)

        scenes = scene_graph.get("scenes", [])
        scene_id = "scene_000"
        optimal_cx = 0.50
        reframe_mode = "auto_reframe"

        for sc in scenes:
            if sc.get("start_time", 0.0) <= source_start <= sc.get("end_time", 0.0):
                scene_id = sc.get("scene_id", scene_id)
                reframe_cfg = sc.get("reframe", {})
                optimal_cx = reframe_cfg.get("optimal_center_x", optimal_cx)
                reframe_mode = reframe_cfg.get("mode", reframe_mode)
                break

        cleaned_text = gemini_raw_plan.get("cleaned_transcript", "").strip()

        narrative_summary = (
            f"Hook: {gemini_raw_plan.get('hook_quote', '')} | "
            f"Punchline: {gemini_raw_plan.get('punchline_conclusion_quote', '')}"
        )

        selected_clip = {
            "clip_id": "clip_01",
            "scene_id": scene_id,
            "source_start": round(source_start, 3),
            "source_end": round(source_end, 3),
            "duration": clip_duration,
            "optimal_center_x": optimal_cx,
            "reframe_mode": reframe_mode,
            "narrative_purpose": narrative_summary,
            "cleaned_transcript": cleaned_text
        }

        hook_duration = min(4.0, clip_duration)
        hook_segment = {
            "source_start": round(source_start, 3),
            "source_end": round(source_start + hook_duration, 3),
            "headline": gemini_raw_plan.get("headline_hook", "Momen Penting Pembuka"),
            "reason": gemini_raw_plan.get("editorial_reasoning", "Hook terkurasi")
        }

        return {
            "status": "success",
            "plan_source": "gemini_api",
            "model_used": successful_model,
            "target_aspect_ratio": "9:16",
            "total_planned_duration": clip_duration,
            "hook_segment": hook_segment,
            "selected_clips": [selected_clip],
            "cleaned_transcript": cleaned_text,
            "editorial_reasoning": gemini_raw_plan.get("editorial_reasoning", "Kurasi tiga babak tuntas.")
        }

    except Exception as parse_err:
        logger.error(f"Gagal memproses hasil pilihan Gemini: {parse_err}. Mengaktifkan fallback lokal.")
        return generate_local_rule_based_plan(scene_graph, target_max_duration)


def run_gemini_planner_pipeline(
    scene_graph_path: Path,
    output_dir: Path,
    model_name: str = "gemini-3.7-flash",
    fallback_model: str = "gemini-3.6-flash",
    target_max_duration: float = 60.0
) -> Dict[str, Any]:
    """
    Menjalankan alur Perencanaan Narasi AI (Tahap 5):
    1. Membaca berkas scene_graph.json
    2. Menjalankan perencanaan editorial dengan proteksi cascading fallback
    3. Menyimpan hasil draf ke draft_plan.json
    """
    scene_graph_path = Path(scene_graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memulai eksekusi Tahap 5 (Perencanaan Narasi AI / Modul Otak)")

    if not scene_graph_path.exists():
        raise FileNotFoundError(f"Berkas Scene Graph tidak ditemukan: {scene_graph_path}")

    with open(scene_graph_path, "r", encoding="utf-8") as f:
        scene_graph = json.load(f)

    plan = plan_editorial_with_gemini(
        scene_graph=scene_graph,
        model_name=model_name,
        fallback_model=fallback_model,
        target_max_duration=target_max_duration
    )

    output_path = output_dir / "draft_plan.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    logger.success(f"Tahap 5 selesai. Draf rencana editorial tersimpan di {output_path.resolve()}")
    return plan