#!/usr/bin/env python3
# Flask UI for realtime-edge-small-object-detection

import os
import json
import uuid
import shlex
import subprocess
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify, Response
)
from werkzeug.utils import secure_filename

# ====== Varsayılan yollar ======
ENGINE_PATH = "/home/ula-sak-n/Ulas/model3_personal_computer.engine"
META_PATH   = "/home/ula-sak-n/Ulas/model1.json"

# Senin video inference scriptlerin
NORFAIR_VIDEO_SCRIPT   = "/home/ula-sak-n/Ulas/video_tracking_norfair.py"
BYTETRACK_VIDEO_SCRIPT = "/home/ula-sak-n/Ulas/video_tracking_bytetrack.py"

# Klasörler
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# UI default paramlar
DEFAULTS = {
    "conf_thres": 0.30,
    "iou_thres":  0.50,
    "nms_mode":   "agnostic",
    # ByteTrack
    "track_thres":  0.60,
    "match_thres":  0.80,
    "track_buffer": 1000,
    "min_box_area": 10.0,
}

# Flask
app = Flask(__name__)
app.secret_key = "change-this-secret"

# === Click-to-track state ===
selected_ids = set()          # kullanıcı seçimleri (ID set)
detections_by_frame = {}      # { int(frame_idx): [ {id,x1,y1,x2,y2}, ... ] }

# -------------------- Helpers --------------------
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
    """GPU codec kalıntısı olmadan h264 + faststart ile yaz."""
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
        env={"LD_LIBRARY_PATH": ""},   # kritik
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{completed.stdout}")

def get_video_fps(video_path, fallback=30.0):
    """ffprobe ile FPS bul; olmazsa fallback."""
    try:
        cmd = [
            "/usr/bin/ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=nk=1:nw=1",
            str(video_path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if "/" in out:
            num, den = out.split("/")
            num, den = float(num), float(den)
            if den != 0:
                return float(num / den)
        val = float(out)
        return val if val > 1 else fallback
    except Exception:
        return fallback

def load_tracks_json(json_path: Path):
    """tracks.json dosyasını okuyup detections_by_frame biçimine çevir."""
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # İki olası formatı da destekle:
    # A) {"0":[{...}], "1":[...], ...}
    # B) [{"frame":0,"id":...,"x1":...}, ...]
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                ik = int(k)
            except Exception:
                continue
            cleaned = []
            for d in v:
                try:
                    cleaned.append({
                        "id": int(d["id"]),
                        "x1": int(d["x1"]), "y1": int(d["y1"]),
                        "x2": int(d["x2"]), "y2": int(d["y2"]),
                    })
                except Exception:
                    continue
            out[ik] = cleaned
    elif isinstance(data, list):
        for d in data:
            try:
                fidx = int(d["frame"])
                item = {
                    "id": int(d["id"]),
                    "x1": int(d["x1"]), "y1": int(d["y1"]),
                    "x2": int(d["x2"]), "y2": int(d["y2"]),
                }
                out.setdefault(fidx, []).append(item)
            except Exception:
                continue
    return out

def try_load_tracks_json(for_video_path: Path):
    """
    Çıktı videonun yanında bir tracks JSON dosyası varsa yükler ve (data, path) döndürür.
    Kabul edilen isim örnekleri:
      *.tracks.json
      *_tracked.tracks.json
      *_tracked.json
      *.json
    """
    candidates = []
    stem = for_video_path.stem
    # *_fixed.mp4 -> *_fixed.tracks.json
    candidates.append(for_video_path.with_suffix(".tracks.json"))
    # *_fixed.mp4 -> *.tracks.json (fixed'siz)
    if stem.endswith("_fixed"):
        candidates.append(for_video_path.with_name(stem.replace("_fixed", "") + ".tracks.json"))
        candidates.append(for_video_path.with_name(stem.replace("_fixed", "") + ".json"))
    # *.mp4 -> *.json
    candidates.append(for_video_path.with_suffix(".json"))

    for cand in candidates:
        if cand.exists():
            print(f"[tracks] loading {cand}")
            return load_tracks_json(cand), cand
    print("[tracks] no tracks JSON found next to output video.")
    return {}, None

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
    render_mode,          # <-- "all" | "none"
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
        "--nms-mode",   nms_mode,
        "--render",     render_mode,  # <-- önemli
    ]
    if tracker_kind == "bytetrack" and bt_params:
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

# ===================== Live camera streaming (TRT + GStreamer) =====================
# Lazım olacak bağımlılıklar:
# ===================== Live camera streaming (TRT + GStreamer) =====================
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# --- Norfair KALDIRILDI; ByteTrack eklendi ---
from yolox.tracker.byte_tracker import BYTETracker
import torch

# Patch for numpy >=1.24 compatibility with ByteTrack
if not hasattr(np, "float"):
    np.float = float

# ---- Global CUDA primary context (create once) ----
cuda.init()
CUDA_DEVICE = cuda.Device(0)
CUDA_CTX = CUDA_DEVICE.retain_primary_context()  # will be pushed in the streaming thread as needed

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def trt_to_np_dtype(trt_dtype: trt.DataType):
    if trt_dtype == trt.DataType.FLOAT:
        return np.float32
    if trt_dtype == trt.DataType.HALF:
        return np.float16
    return np.float32

def letterbox(im, new_shape, color=(114, 114, 114)):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw, dh = (new_w - nw) // 2, (new_h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh

def nms_xyxy_class_agnostic(xyxy, scores, iou_thres=0.5):
    if len(xyxy) == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = xyxy[:,0], xyxy[:,1], xyxy[:,2], xyxy[:,3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0); h = (yy2 - yy1).clip(0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

def run_nms(xyxy, conf, cls_ids, iou_thres=0.5):
    if xyxy.size == 0:
        return xyxy, conf, cls_ids
    keep = nms_xyxy_class_agnostic(xyxy, conf, iou_thres)
    return xyxy[keep], conf[keep], cls_ids[keep]

def _xywh_to_xyxy(xywh):
    cx, cy, w, h = xywh
    return cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

def _load_engine_runtime(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        eng = rt.deserialize_cuda_engine(f.read())
    if eng is None:
        raise RuntimeError("Failed to deserialize engine")
    return eng

def _open_camera_with_fallback(cfg):
    # simplified: just open /dev/video
    cap = cv2.VideoCapture(cfg["device"])
    return cap

def _gen_camera_frames():
    global detections_by_frame
    try:
        CUDA_CTX.push()
    except Exception as e:
        print(f"[cuda] push failed: {e}")
        raise

    try:
        cfg = CAMERA_CFG.copy()
        with open(cfg["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        img_size = meta.get("img_size", [640, 640])
        in_w, in_h = (img_size[0], img_size[1]) if len(img_size) == 2 else (640, 640)
        input_name  = meta.get("input_name")
        output_name = meta.get("output_name")

        engine = _load_engine_runtime(cfg["engine"])
        ctx = engine.create_execution_context()
        if input_name  is None: input_name  = engine.get_tensor_name(0)
        if output_name is None: output_name = engine.get_tensor_name(1)
        ctx.set_input_shape(input_name, (1, 3, in_h, in_w))

        in_dtype  = trt_to_np_dtype(engine.get_tensor_dtype(input_name))
        out_dtype = trt_to_np_dtype(engine.get_tensor_dtype(output_name))
        in_shape  = ctx.get_tensor_shape(input_name)
        out_shape = ctx.get_tensor_shape(output_name)

        n_input  = int(np.prod(in_shape))
        n_output = int(np.prod(out_shape))
        d_in  = cuda.mem_alloc(n_input  * np.dtype(in_dtype).itemsize)
        d_out = cuda.mem_alloc(n_output * np.dtype(out_dtype).itemsize)
        h_in  = np.empty(n_input,  dtype=in_dtype)
        h_out = np.empty(n_output, dtype=out_dtype)
        stream = cuda.Stream()

        cap = _open_camera_with_fallback(cfg)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")

        class _BTArgs:
            track_thresh = DEFAULTS.get("track_thres", 0.6)
            match_thresh = DEFAULTS.get("match_thres", 0.8)
            track_buffer = DEFAULTS.get("track_buffer", 1000)
            min_box_area = DEFAULTS.get("min_box_area", 10.0)
            mot20 = False
            frame_rate = cfg.get("fps", 30)

        tracker = BYTETracker(_BTArgs())
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # preprocess
            lb_img, r, dw, dh = letterbox(frame, (in_w, in_h))
            img = lb_img.astype(np.float32).transpose(2, 0, 1)[None, ...]
            if in_dtype == np.float16:
                img = img.astype(np.float16)
            np.copyto(h_in, img.ravel())

            # TRT inference
            cuda.memcpy_htod_async(d_in, h_in, stream)
            ctx.set_tensor_address(input_name, int(d_in))
            ctx.set_tensor_address(output_name, int(d_out))
            ctx.execute_async_v3(stream.handle)
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
            stream.synchronize()

            out = h_out.astype(np.float32).reshape(out_shape)
            out = np.squeeze(out, axis=0)

            if out.shape[1] == 6:
                xyxy = out[:, :4].astype(np.float32)
                conf = out[:, 4].astype(np.float32)
            else:
                nc = out.shape[1] - 5
                cxcywh = out[:, 0:4].astype(np.float32)
                obj    = out[:, 4:5].astype(np.float32)
                cls_p  = out[:, 5:].astype(np.float32)
                scores_all = obj * cls_p
                conf = scores_all.max(axis=1)
                keep = conf >= cfg["conf_thres"]
                cxcywh, conf = cxcywh[keep], conf[keep]
                xyxy = np.array([_xywh_to_xyxy(b) for b in cxcywh], dtype=np.float32)
                xyxy, conf, _ = run_nms(xyxy, conf, np.zeros_like(conf), cfg["iou_thres"])

            dets = []
            if xyxy.size:
                h0, w0 = frame.shape[:2]
                xyxy[:, [0, 2]] -= dw
                xyxy[:, [1, 3]] -= dh
                xyxy[:, :4] /= r
                xyxy = np.clip(xyxy, 0, [w0 - 1, h0 - 1, w0 - 1, h0 - 1])
                for (x1, y1, x2, y2), sc in zip(xyxy, conf):
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(sc)])

            dets_bt = torch.tensor(dets, dtype=torch.float32, device="cpu") if dets else torch.empty((0, 5), dtype=torch.float32, device="cpu")

            img_h, img_w = frame.shape[:2]
            img_info = (img_h, img_w)
            img_size = (in_h, in_w)

            online_targets = tracker.update(dets_bt, img_info, img_size)

            # save detections for click tracker
            det_list = []
            for t in online_targets:
                x1, y1, x2, y2 = map(int, t.tlbr)
                det_list.append({"id": int(t.track_id), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            detections_by_frame[frame_idx] = det_list

            # draw
            mode = CAMERA_CFG.get("tracking_mode", "auto")
            
            for t in online_targets:
                tid = int(t.track_id)
            
                if mode == "interactive":
                    # Eğer seçili ID yoksa → hiçbir şey çizme
                    if not selected_ids:
                        continue
                    # Eğer seçili ID var ama bu ID seçili listede değilse → çizme
                    if tid not in selected_ids:
                        continue
            
                # AUTO modda her zaman çizilir
                x1, y1, x2, y2 = map(int, t.tlbr)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

            frame_idx += 1

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(buf) + b"\r\n")

        cap.release()
        try:
            d_in.free(); d_out.free()
        except Exception:
            pass
    finally:
        try:
            CUDA_CTX.pop()
        except Exception as e:
            print(f"[cuda] pop failed: {e}")

# -------------------- Routes --------------------
CAMERA_CFG = {
    "engine": ENGINE_PATH,
    "meta": META_PATH,
    "device": "/dev/video2",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "camera": "usb",
    "conf_thres": DEFAULTS["conf_thres"],
    "iou_thres": DEFAULTS["iou_thres"],
    "nms_mode": DEFAULTS["nms_mode"],
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mode          = request.form.get("mode")              # upload | camera
        tracker       = request.form.get("tracker")           # norfair | bytetrack
        tracking_mode = request.form.get("tracking_mode", "auto")

        engine_in = (request.form.get("engine_path") or "").strip()
        meta_in   = (request.form.get("meta_path")   or "").strip()
        effective_engine = engine_in if engine_in else ENGINE_PATH
        effective_meta   = meta_in   if meta_in   else META_PATH

        conf_thres = _to_float(request.form.get("conf_thres"), DEFAULTS["conf_thres"])
        iou_thres  = _to_float(request.form.get("iou_thres"),  DEFAULTS["iou_thres"])
        nms_mode   = request.form.get("nms_mode") or DEFAULTS["nms_mode"]
        if nms_mode not in ("agnostic", "aware"):
            nms_mode = DEFAULTS["nms_mode"]

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

            selected_ids.clear()
            detections_by_frame.clear()

            try:
                render_flag = "all" if tracking_mode == "auto" else "none"

                if tracker == "norfair":
                    run_tracking(
                        NORFAIR_VIDEO_SCRIPT, in_path, out_path, "norfair",
                        effective_engine, effective_meta,
                        conf_thres, iou_thres, nms_mode,
                        render_flag, None
                    )
                elif tracker == "bytetrack":
                    run_tracking(
                        BYTETRACK_VIDEO_SCRIPT, in_path, out_path, "bytetrack",
                        effective_engine, effective_meta,
                        conf_thres, iou_thres, nms_mode,
                        render_flag, bt_params
                    )
                else:
                    flash("Geçersiz tracker seçimi.")
                    return redirect(url_for("index"))

                fixed_out = OUTPUT_DIR / f"{stem}_{uid}_tracked_fixed.mp4"
                fix_mp4_codec(out_path, fixed_out)
                out_path = fixed_out
                out_name = out_path.name

                loaded, tracks_path = try_load_tracks_json(out_path)
                tracks_name = ""
                if loaded:
                    detections_by_frame.update(loaded)
                    tracks_name = tracks_path.name if tracks_path else ""
                fps_val = get_video_fps(out_path, fallback=30.0)

            except Exception as e:
                flash(f"Inference hatası: {e}")
                return redirect(url_for("index"))

            return redirect(url_for(
                "result",
                in_video=in_name,
                out_video=out_name,
                tracker=tracker,
                conf=conf_thres, iou=iou_thres, nms=nms_mode,
                tracking_mode=tracking_mode,
                fps=fps_val,
                tracks=tracks_name
            ))

        elif mode == "camera":
            CAMERA_CFG.update({
                "engine": effective_engine,
                "meta":   effective_meta,
                "conf_thres": conf_thres,
                "iou_thres":  iou_thres,
                "nms_mode":   nms_mode,
                "device": "/dev/video2",
                "width": 1280, "height": 720, "fps": 30,
                "camera": "usb",
                "tracking_mode": tracking_mode
            })
            selected_ids.clear()
            detections_by_frame.clear()

            return render_template(
                "result.html",
                is_live=True,
                tracker=tracker,
                conf=conf_thres,
                iou=iou_thres,
                nms=nms_mode,
                tracking_mode=tracking_mode
            )

        else:
            flash("Lütfen bir mod seçiniz (Video Yükle / Kamera).")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        default_engine=ENGINE_PATH,
        default_meta=META_PATH,
        defaults=DEFAULTS
    )

@app.route("/result")
def result():
    in_video      = request.args.get("in_video")
    out_video     = request.args.get("out_video")
    tracker       = request.args.get("tracker", "norfair")
    tracking_mode = request.args.get("tracking_mode", "auto")
    fps           = float(request.args.get("fps", 30))
    tracks_name   = request.args.get("tracks", "")
    is_live       = request.args.get("is_live", "false").lower() == "true"

    if not is_live and (not in_video or not out_video):
        return redirect(url_for("index"))

    tracks_url = url_for("serve_output", filename=tracks_name) if tracks_name else ""

    return render_template(
        "result.html",
        is_live=is_live,
        in_video=in_video,
        out_video=out_video,
        tracker=tracker,
        conf=request.args.get("conf"),
        iou=request.args.get("iou"),
        nms=request.args.get("nms"),
        tracking_mode=tracking_mode,
        fps=fps,
        tracks_url=tracks_url
    )

@app.route("/camera_feed")
def camera_feed():
    return Response(_gen_camera_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --------- Static serving ----------
@app.route("/static/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# --------- Legal / About ----------
@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/select_id", methods=["POST"])
def select_id():
    data  = request.get_json() or {}
    x     = data.get("x")
    y     = data.get("y")
    frame = int(data.get("frame", 0))

    print(f"[select_id] click at ({x}, {y}), frame={frame}")
    print(f"[select_id] detections_by_frame keys: {list(detections_by_frame.keys())[:10]} ...")
    print(f"[select_id] currently selected_ids: {selected_ids}")

    if x is None or y is None:
        return jsonify({"status": "error", "msg": "x,y required", "selected_ids": list(selected_ids)}), 400

    # Kamera modunda frame uyuşmayabilir → en son frame kullan
    if frame not in detections_by_frame and detections_by_frame:
        frame = max(detections_by_frame.keys())
        print(f"[select_id] fallback to last frame={frame}")

    dets = detections_by_frame.get(frame, [])
    print(f"[select_id] detections in this frame: {dets}")

    if not dets:
        return jsonify({"status": "no-data", "msg": "no detections for this frame", "selected_ids": list(selected_ids)})

    PAD = 4
    chosen_id = None
    best_rank = None

    for d in dets:
        x1 = d["x1"] - PAD
        y1 = d["y1"] - PAD
        x2 = d["x2"] + PAD
        y2 = d["y2"] + PAD
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            rank = 1.0 / area
            if best_rank is None or rank > best_rank:
                best_rank = rank
                chosen_id = int(d["id"])

    print(f"[select_id] chosen_id={chosen_id}")

    if chosen_id is not None:
        if chosen_id in selected_ids:
            selected_ids.remove(chosen_id)
            action = "removed"
        else:
            selected_ids.add(chosen_id)
            action = "added"
        print(f"[select_id] updated selected_ids={selected_ids}")
        return jsonify({"status": "ok", "id": chosen_id, "action": action, "selected_ids": list(selected_ids)})
    else:
        return jsonify({"status": "miss", "msg": "click not on any box", "selected_ids": list(selected_ids)})



@app.route("/clear_selected", methods=["POST"])
def clear_selected():
    selected_ids.clear()
    return jsonify({"status": "ok", "selected_ids": []})

@app.route("/frame_dets")
def frame_dets():
    try:
        frame = int(request.args.get("frame", 0))
    except Exception:
        frame = 0
    return jsonify(detections_by_frame.get(frame, []))

@app.route("/selected_ids")
def get_selected_ids():
    return jsonify(sorted(list(selected_ids)))

# --------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        try:
            CUDA_CTX.detach()
        except Exception:
            pass

