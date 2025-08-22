#!/usr/bin/env python3
# Flask UI for realtime-edge-small-object-detection
import os
import uuid
import shlex
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# ====== Varsayılan yollar (boş formda bunlar kullanılır) ======
ENGINE_PATH = "/home/ula-sak-n/Ulas/model3_personal_computer.engine"
META_PATH   = "/home/ula-sak-n/Ulas/model1.json"

# Senin video inference scriptlerin:
NORFAIR_VIDEO_SCRIPT   = "/home/ula-sak-n/Ulas/video_tracking_norfair.py"    # --video --save ... destekliyor :contentReference[oaicite:3]{index=3}
BYTETRACK_VIDEO_SCRIPT = "/home/ula-sak-n/Ulas/video_tracking_bytetrack.py"  # ByteTrack args'ları da var :contentReference[oaicite:4]{index=4}

# Klasörler
BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# UI default paramlar
DEFAULTS = {
    "conf_thres": 0.30,
    "iou_thres":  0.50,
    "nms_mode":   "agnostic",   # or "aware"
    # ByteTrack
    "track_thres":  0.60,
    "match_thres":  0.80,
    "track_buffer": 1000,
    "min_box_area": 10.0,
}

# Flask
app = Flask(__name__)
app.secret_key = "change-this-secret"

def _to_float(v, fallback):
    try:
        return float(v)
    except Exception:
        return fallback

def _to_int(v, fallback):
    try:
        return int(v)
    except Exception:
        return fallback

def fix_mp4_codec(input_path, output_path):
    cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-movflags", "+faststart",
        str(output_path)
    ]
    completed = subprocess.run(
        cmd,
        env={"LD_LIBRARY_PATH": ""},   # <-- kritik
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{completed.stdout}")




def run_tracking(
    script,
    input_video_path,
    output_video_path,
    tracker_kind,
    engine_path,
    meta_path,
    conf_thres,
    iou_thres,
    nms_mode,
    bt_params=None
):
    """Verilen scripti subprocess ile çağırır."""
    cmd = [
        "python", str(script),
        "--engine", str(engine_path),
        "--meta",   str(meta_path),
        "--video",  str(input_video_path),
        "--save",   str(output_video_path),
        "--conf-thres", f"{conf_thres}",
        "--iou-thres",  f"{iou_thres}",
        "--nms-mode",   nms_mode,  # "agnostic" | "aware"
    ]

    if tracker_kind == "bytetrack" and bt_params:
        # ByteTrack CLI arg'ları scriptinde var :contentReference[oaicite:5]{index=5}
        cmd += [
            "--track-thres",  f"{bt_params['track_thres']}",
            "--match-thres",  f"{bt_params['match_thres']}",
            "--track-buffer", f"{bt_params['track_buffer']}",
            "--min-box-area", f"{bt_params['min_box_area']}",
        ]

    print(">>> RUN:", " ".join(shlex.quote(x) for x in cmd))
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Inference process failed (rc={completed.returncode}). See logs above.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Mod & tracker
        mode    = request.form.get("mode")            # "upload" | "camera" (camera'yı sonraki adımda ekleyeceğiz)
        tracker = request.form.get("tracker")         # "norfair" | "bytetrack"

        # Engine/meta — boşsa default
        engine_in = (request.form.get("engine_path") or "").strip()
        meta_in   = (request.form.get("meta_path")   or "").strip()
        effective_engine = engine_in if engine_in else ENGINE_PATH
        effective_meta   = meta_in   if meta_in   else META_PATH

        # Ortak postprocess parametreleri
        conf_thres = _to_float(request.form.get("conf_thres"), DEFAULTS["conf_thres"])
        iou_thres  = _to_float(request.form.get("iou_thres"),  DEFAULTS["iou_thres"])
        nms_mode   = request.form.get("nms_mode") or DEFAULTS["nms_mode"]
        if nms_mode not in ("agnostic", "aware"):
            nms_mode = DEFAULTS["nms_mode"]

        # ByteTrack ek parametreleri (seçilirse)
        bt_params = None
        if tracker == "bytetrack":
            bt_params = {
                "track_thres":  _to_float(request.form.get("track_thres"),  DEFAULTS["track_thres"]),
                "match_thres":  _to_float(request.form.get("match_thres"),  DEFAULTS["match_thres"]),
                "track_buffer": _to_int  (request.form.get("track_buffer"), DEFAULTS["track_buffer"]),
                "min_box_area": _to_float(request.form.get("min_box_area"), DEFAULTS["min_box_area"]),
            }

        if mode == "upload":
            f = request.files.get("video_file")
            if not f or f.filename == "":
                flash("Lütfen bir video dosyası yükleyin.")
                return redirect(url_for("index"))

            filename = secure_filename(f.filename)
            stem = Path(filename).stem
            ext  = Path(filename).suffix.lower()
            uid  = uuid.uuid4().hex[:8]
            in_name  = f"{stem}_{uid}{ext}"
            out_name = f"{stem}_{uid}_tracked.mp4"

            in_path  = UPLOAD_DIR / in_name
            out_path = OUTPUT_DIR / out_name
            f.save(str(in_path))

            try:
                if tracker == "norfair":
                    # Norfair scripti: --video/--save/--conf-thres/--iou-thres/--nms-mode destekli :contentReference[oaicite:6]{index=6}
                    run_tracking(
                        NORFAIR_VIDEO_SCRIPT, in_path, out_path, "norfair",
                        effective_engine, effective_meta,
                        conf_thres, iou_thres, nms_mode, None
                    )
                elif tracker == "bytetrack":
                    # ByteTrack scripti: yukarıdakilere ek olarak BT hiperparametreleri destekli :contentReference[oaicite:7]{index=7}
                    run_tracking(
                        BYTETRACK_VIDEO_SCRIPT, in_path, out_path, "bytetrack",
                        effective_engine, effective_meta,
                        conf_thres, iou_thres, nms_mode, bt_params
                    )
                else:
                    flash("Geçersiz tracker seçimi.")
                    return redirect(url_for("index"))

                
                
                fixed_out = OUTPUT_DIR / f"{stem}_{uid}_tracked_fixed.mp4"
                fix_mp4_codec(out_path, fixed_out)
                out_path = fixed_out
                out_name = out_path.name
            except Exception as e:
                flash(f"Inference hatası: {e}")
                return redirect(url_for("index"))

            # Sonuç sayfasına paramları da taşıyalım (gösterim için opsiyonel)
            return redirect(url_for(
                "result",
                in_video=in_name,
                out_video=out_name,
                tracker=tracker,
                conf=conf_thres, iou=iou_thres, nms=nms_mode
            ))

        elif mode == "camera":
            flash("Kamera (real-time) akışı web'e aktarma adımını hemen ardından ekleyelim (MJPEG/HLS). Şu an video yükleme akışı hazır.")
            return redirect(url_for("index"))

        else:
            flash("Lütfen bir mod seçiniz (Video Yükle / Kamera).")
            return redirect(url_for("index"))

    # GET — form default değerleri
    return render_template(
        "index.html",
        default_engine=ENGINE_PATH,
        default_meta=META_PATH,
        defaults=DEFAULTS
    )

@app.route("/result")
def result():
    in_video  = request.args.get("in_video")
    out_video = request.args.get("out_video")
    tracker   = request.args.get("tracker", "norfair")
    if not in_video or not out_video:
        return redirect(url_for("index"))
    return render_template(
        "result.html",
        in_video=in_video,
        out_video=out_video,
        tracker=tracker,
        conf=request.args.get("conf"),
        iou=request.args.get("iou"),
        nms=request.args.get("nms")
    )

@app.route("/static/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
