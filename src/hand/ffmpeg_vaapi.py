import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


FFMPEG_QUIET = ["-hide_banner", "-loglevel", "error", "-nostats"]
AUDIO_FADE_SECONDS = 0.008


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Menjalankan perintah FFmpeg dan menangkap keluaran galatnya."""
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None
    )


def has_audio_stream(video_path: Path) -> bool:
    """Memeriksa apakah berkas video memiliki stream audio."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() != ""


def probe_duration(video_path: Path) -> float:
    """Membaca durasi berkas media dalam detik."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def build_audio_filter(duration: float) -> str:
    """Fade sangat singkat di awal dan akhir klip agar tidak ada bunyi 'tek' di sambungan."""
    fade_out_start = max(0.0, duration - AUDIO_FADE_SECONDS)
    return (
        f"afade=t=in:st=0:d={AUDIO_FADE_SECONDS},"
        f"afade=t=out:st={fade_out_start:.6f}:d={AUDIO_FADE_SECONDS}"
    )


def build_segment_commands(
    seg: Dict[str, Any],
    source_video: Path,
    segment_path: Path,
    fps: int,
    out_w: int,
    out_h: int,
    with_audio: bool,
    vaapi_device: str,
    qp: int,
    cpu_crf: int
) -> Dict[str, List[str]]:
    """
    Membuat dua versi perintah untuk satu klip: jalur hardware VAAPI dan jalur
    cadangan CPU. Crop dilakukan di CPU (hanya pointer, sangat murah), sedangkan
    upscale ke 1080x1920 dikerjakan iGPU lewat scale_vaapi.
    Hasil antara memakai audio PCM agar durasi audio dan video presisi dan
    tidak ada penundaan encoder AAC di tiap sambungan klip.
    """
    crop = seg["crop"]
    crop_filter = f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"
    start_sec = f"{seg['source_start_sec']:.6f}"
    duration = seg["duration_sec"]
    dur_sec = f"{duration:.6f}"

    audio_args: List[str]
    if with_audio:
        audio_args = [
            "-af", build_audio_filter(duration),
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"
        ]
    else:
        audio_args = ["-an"]

    common_in = ["-ss", start_sec, "-i", str(source_video), "-t", dur_sec]
    common_out = ["-fps_mode", "cfr", "-r", str(fps)]

    cmd_vaapi = (
        ["ffmpeg", "-y"] + FFMPEG_QUIET
        + ["-vaapi_device", vaapi_device]
        + common_in
        + ["-vf", f"{crop_filter},format=nv12,hwupload,scale_vaapi=w={out_w}:h={out_h}"]
        + ["-c:v", "h264_vaapi", "-qp", str(qp)]
        + common_out + audio_args
        + [str(segment_path)]
    )

    cmd_cpu = (
        ["ffmpeg", "-y"] + FFMPEG_QUIET
        + common_in
        + ["-vf", f"{crop_filter},scale={out_w}:{out_h}:flags=bicubic,setsar=1"]
        + ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cpu_crf), "-pix_fmt", "yuv420p"]
        + common_out + audio_args
        + [str(segment_path)]
    )

    return {"vaapi": cmd_vaapi, "cpu": cmd_cpu}


def execute_with_fallback(
    cmd_vaapi: List[str],
    cmd_cpu: List[str],
    state: Dict[str, Any],
    label: str,
    cwd: Optional[Path] = None
) -> None:
    """
    Mencoba hardware VAAPI dulu. Jika gagal, beralih ke CPU libx264 dan
    mematikan VAAPI untuk sisa proses agar tidak gagal berulang-ulang.
    """
    if state["use_hw"]:
        res = run_command(cmd_vaapi, cwd=cwd)
        if res.returncode == 0:
            state["hw_jobs"] += 1
            return
        logger.warning(f"[{label}] VAAPI gagal: {res.stderr.strip()[-300:]}")
        logger.warning("Beralih ke mode fallback: CPU encoder libx264 untuk sisa proses...")
        state["use_hw"] = False

    res = run_command(cmd_cpu, cwd=cwd)
    if res.returncode != 0:
        logger.error(f"[{label}] Encode CPU gagal: {res.stderr.strip()[-500:]}")
        raise RuntimeError(f"Gagal merender {label}: {res.stderr.strip()[-500:]}")
    state["cpu_jobs"] += 1


def write_concat_list(segment_paths: List[Path], list_path: Path) -> None:
    """Menulis daftar berkas untuk concat demuxer FFmpeg."""
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            safe = str(Path(p).resolve()).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")


def render_final_pass(
    concat_list: Path,
    output_path: Path,
    subtitle_path: Optional[Path],
    with_audio: bool,
    vaapi_device: str,
    qp: int,
    cpu_crf: int,
    state: Dict[str, Any],
    work_dir: Path
) -> None:
    """
    Tahap akhir: menyambung semua klip, membakar subtitle (jika ada), meredam
    kliping puncak suara via alimiter (0.85 = -1.4 dBFS), dan encode final ke MP4 H.264 + AAC.
    """
    # Batas 0.85 memberi ruang headroom terhadap lonjakan codec AAC sehingga volume puncak tertahan di bawah 0.0 dB
    audio_args = [
        "-af", "alimiter=limit=0.85",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000"
    ] if with_audio else ["-an"]

    base_in = ["ffmpeg", "-y"] + FFMPEG_QUIET + ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
    tail = ["-movflags", "+faststart", str(output_path.resolve())]

    if subtitle_path is None:
        cmd = base_in + ["-c:v", "copy"] + audio_args + tail
        res = run_command(cmd, cwd=work_dir)
        if res.returncode != 0:
            raise RuntimeError(f"Gagal menyambung klip: {res.stderr.strip()[-500:]}")
        return

    sub_name = Path(subtitle_path).name
    if not (work_dir / sub_name).exists():
        shutil.copy2(subtitle_path, work_dir / sub_name)

    cmd_vaapi = (
        base_in
        + ["-vaapi_device", vaapi_device]
        + ["-vf", f"ass={sub_name},format=nv12,hwupload"]
        + ["-c:v", "h264_vaapi", "-qp", str(qp)]
        + ["-fps_mode", "cfr"]
        + audio_args + tail
    )
    cmd_cpu = (
        base_in
        + ["-vf", f"ass={sub_name}"]
        + ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cpu_crf), "-pix_fmt", "yuv420p"]
        + ["-fps_mode", "cfr"]
        + audio_args + tail
    )
    execute_with_fallback(cmd_vaapi, cmd_cpu, state, "render akhir + subtitle", cwd=work_dir)


def run_ffmpeg_render_pipeline(
    manifest: Dict[str, Any],
    output_dir: Path,
    subtitle_path: Optional[Path] = None,
    output_name: str = "final_render.mp4",
    use_hardware: bool = True,
    vaapi_device: str = "/dev/dri/renderD128",
    segment_qp: int = 18,
    final_qp: int = 21,
    cpu_crf: int = 20,
    keep_temp: bool = False
) -> Dict[str, Any]:
    """
    Menjalankan render fisik (Tahap 7 bagian 3):
    1. Merender tiap klip: seek cepat, crop 9:16, upscale di iGPU (VAAPI)
    2. Menyambung seluruh klip
    3. Membakar subtitle, normalisasi puncak volume, dan encode final H.264 + AAC
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_video = Path(manifest["source_video"])
    if not source_video.exists():
        raise FileNotFoundError(f"Video sumber tidak ditemukan: {source_video}")

    fps = int(manifest["fps"])
    out_w = int(manifest["output_width"])
    out_h = int(manifest["output_height"])
    segments = manifest["segments"]
    if not segments:
        raise ValueError("Manifes render tidak memiliki segmen.")

    hw_available = use_hardware and Path(vaapi_device).exists()
    if use_hardware and not hw_available:
        logger.warning(f"Perangkat VAAPI {vaapi_device} tidak ditemukan. Memakai CPU.")

    state = {"use_hw": hw_available, "hw_jobs": 0, "cpu_jobs": 0}
    with_audio = has_audio_stream(source_video)
    if not with_audio:
        logger.warning("Video sumber tidak memiliki audio. Render dilakukan tanpa suara.")

    logger.info("Memulai eksekusi Tahap 7C (Render FFmpeg VAAPI / Modul Tangan)")
    started = time.time()

    seg_dir = output_dir / "_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    segment_paths: List[Path] = []

    # 1. Render klip satu per satu
    for seg in segments:
        seg_path = seg_dir / f"seg_{seg['index']:03d}.mkv"
        cmds = build_segment_commands(
            seg, source_video, seg_path, fps, out_w, out_h,
            with_audio, vaapi_device, segment_qp, cpu_crf
        )
        label = f"klip {seg['index']}/{len(segments)} ({seg['clip_id']})"
        logger.info(f"Merender {label}: {seg['duration_sec']}s")
        execute_with_fallback(cmds["vaapi"], cmds["cpu"], state, label)
        segment_paths.append(seg_path)

    # 2. Daftar sambungan
    concat_list = seg_dir / "concat_list.txt"
    write_concat_list(segment_paths, concat_list)

    # 3. Render akhir
    final_path = output_dir / output_name
    sub_path = Path(subtitle_path) if subtitle_path and Path(subtitle_path).exists() else None
    if subtitle_path and sub_path is None:
        logger.warning(f"Subtitle {subtitle_path} tidak ditemukan. Render tanpa subtitle.")

    render_final_pass(
        concat_list, final_path, sub_path, with_audio,
        vaapi_device, final_qp, cpu_crf, state, output_dir
    )

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(f"Render selesai tetapi berkas keluaran tidak valid: {final_path}")

    actual = probe_duration(final_path)
    expected = float(manifest["total_duration"])
    if abs(actual - expected) > 0.25:
        logger.warning(
            f"Durasi hasil render {actual:.2f}s menyimpang dari rencana {expected:.2f}s."
        )

    if not keep_temp:
        shutil.rmtree(seg_dir, ignore_errors=True)

    elapsed = round(time.time() - started, 1)
    result = {
        "output_path": str(final_path.resolve()),
        "duration_seconds": round(actual, 3),
        "expected_duration": expected,
        "hardware_jobs": state["hw_jobs"],
        "cpu_jobs": state["cpu_jobs"],
        "render_seconds": elapsed
    }

    result_path = output_dir / "render_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Tahap 7C selesai dalam {elapsed}s "
        f"(VAAPI: {state['hw_jobs']} job, CPU: {state['cpu_jobs']} job). "
        f"Video: {final_path.resolve()}"
    )
    return result