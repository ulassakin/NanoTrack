#!/usr/bin/env python3
from collections import defaultdict
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import time
np.float = float
np.int = int

# --- ByteTrack ---
from yolox.tracker.byte_tracker import BYTETracker
import torch

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser("EdgeYOLO TensorRT 10.13 + Video (MP4) — show & save")
    ap.add_argument("--engine", required=True, type=str, help="TensorRT engine (.engine)")
    ap.add_argument("--meta", default="", type=str, help="sidecar JSON (auto-detect if empty)")
    ap.add_argument("--video", required=True, type=str, help="Input video path (e.g., input.mp4)")
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=720, type=int)
    ap.add_argument("--fps", default=30, type=int)
    ap.add_argument("--conf-thres", default=0.30, type=float)
    ap.add_argument("--iou-thres", default=0.50, type=float)
    ap.add_argument("--nms-mode", choices=["agnostic", "aware"], default="agnostic")
    ap.add_argument("--save", default="output_tracking_bytetrack1.mp4", type=str, help="Output video path")

    # ByteTrack hiperparametreleri
    ap.add_argument("--track-thres", default=0.6, type=float, help="BYTETracker track_thresh")
    ap.add_argument("--match-thres", default=0.8, type=float, help="BYTETracker match_thresh")
    ap.add_argument("--track-buffer", default=1000, type=int, help="BYTETracker track_buffer (frames)")
    ap.add_argument("--min-box-area", default=10.0, type=float, help="Min box area to keep")

    # YENİ: render modu
    ap.add_argument("--render", choices=["none", "all"], default="none",
                    help="none: save clean video (no boxes), all: draw all tracks on video")
    return ap.parse_args()

# ---------- TRT utils ----------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        eng = rt.deserialize_cuda_engine(f.read())
    if eng is None:
        raise RuntimeError("Failed to deserialize engine.")
    return eng

def trt_to_np_dtype(trt_dtype: trt.DataType):
    if trt_dtype == trt.DataType.FLOAT:
        return np.float32
    if trt_dtype == trt.DataType.HALF:
        return np.float16
    raise RuntimeError(f"Unsupported TRT dtype: {trt_dtype}")

# ---------- meta ----------
def find_meta(engine_path, meta_cli):
    if meta_cli:
        return meta_cli
    p = Path(engine_path)
    cand = p.with_suffix(".json")
    if cand.exists():
        return str(cand)
    for j in p.parent.glob("*.json"):
        return str(j)
    return ""

# ---------- CV helpers ----------
def letterbox(im, new_shape, color=(114, 114, 114)):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw, dh = 0, 0
    canvas[0:nh, 0:nw] = resized
    return canvas, r, dw, dh

def color_for_class(cid: int):
    np.random.seed(cid + 12345)
    c = np.random.randint(60, 255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])

def xywh_to_xyxy(xywh):
    cx, cy, w, h = xywh
    return cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

# ---------- NMS ----------
def nms_xyxy_class_agnostic(xyxy, scores, iou_thres=0.5):
    if len(xyxy) == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
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
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

def run_nms(xyxy, conf, cls_ids, iou_thres=0.5, mode="agnostic"):
    if xyxy.size == 0:
        return xyxy, conf, cls_ids
    if mode == "agnostic":
        keep = nms_xyxy_class_agnostic(xyxy, conf, iou_thres)
        return xyxy[keep], conf[keep], cls_ids[keep]
    # class-aware NMS
    final_boxes, final_scores, final_cids = [], [], []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep_idx = nms_xyxy_class_agnostic(xyxy[idx], conf[idx], iou_thres)
        pick = idx[keep_idx]
        final_boxes.append(xyxy[pick])
        final_scores.append(conf[pick])
        final_cids.append(np.full(len(pick), c, dtype=int))
    if final_boxes:
        return (np.concatenate(final_boxes),
                np.concatenate(final_scores),
                np.concatenate(final_cids))
    return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,))

# ---------- ByteTrack drawing ----------
def draw_tracks(frame, tracks, names=None):
    for t in tracks:
        x1, y1, x2, y2 = t.tlbr
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (36, 255, 12),
            2,
            cv2.LINE_AA,
        )

# ---------- main ----------
def main():
    args = parse_args()

    # TRT
    engine = load_engine(args.engine)
    ctx = engine.create_execution_context()
    stream = cuda.Stream()

    # meta
    meta_path = find_meta(args.engine, args.meta)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    names = meta.get("names", [])
    in_w, in_h = meta.get("img_size", [640, 640])

    input_name  = meta.get("input_name",  engine.get_tensor_name(0))
    output_name = meta.get("output_name", engine.get_tensor_name(1))

    ctx.set_input_shape(input_name, (1, 3, in_h, in_w))
    in_shape  = ctx.get_tensor_shape(input_name)
    out_shape = ctx.get_tensor_shape(output_name)

    in_dtype  = trt_to_np_dtype(engine.get_tensor_dtype(input_name))
    out_dtype = trt_to_np_dtype(engine.get_tensor_dtype(output_name))

    n_input  = int(np.prod(in_shape))
    n_output = int(np.prod(out_shape))

    d_in  = cuda.mem_alloc(n_input  * np.dtype(in_dtype).itemsize)
    d_out = cuda.mem_alloc(n_output * np.dtype(out_dtype).itemsize)
    h_in  = np.empty(n_input,  dtype=in_dtype)
    h_out = np.empty(n_output, dtype=out_dtype)

    # Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    out_writer = None
    out_path = Path(args.save if args.save else "output_camera.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    if cap_fps is None or cap_fps <= 1 or np.isnan(cap_fps):
        cap_fps = float(args.fps)

    # BYTETracker init
    class _BTArgs:
        track_thresh = args.track_thres
        match_thresh = args.match_thres
        track_buffer = args.track_buffer
        mot20 = False
        frame_rate = cap_fps
    tracker = BYTETracker(_BTArgs())

    win_name = "TRT10 Video (press Q/ESC to quit)"

    # --- JSON toplama yapısı ---
    tracks_out = {}  # {frame_idx: [ {id,x1,y1,x2,y2}, ... ] }
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Preprocess
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

        # Postprocess
        if out.shape[1] == 6:
            xyxy, conf, cls_ids = out[:, :4], out[:, 4], out[:, 5].astype(np.int32)
        else:
            nc = out.shape[1] - 5
            cxcywh, obj, cls_prob = out[:, 0:4], out[:, 4:5], out[:, 5:]
            scores_all = obj * cls_prob
            cls_ids = np.argmax(scores_all, 1)
            conf = scores_all[np.arange(out.shape[0]), cls_ids]
            keep = conf >= args.conf_thres
            cxcywh, conf, cls_ids = cxcywh[keep], conf[keep], cls_ids[keep]
            xyxy = np.array([xywh_to_xyxy(b) for b in cxcywh], dtype=np.float32)
            xyxy, conf, cls_ids = run_nms(xyxy, conf, cls_ids, args.iou_thres, args.nms_mode)

        # ByteTrack input
        dets_for_bt = []
        if xyxy.size:
            for i in range(len(xyxy)):
                dets_for_bt.append([
                    float(xyxy[i][0]), float(xyxy[i][1]),
                    float(xyxy[i][2]), float(xyxy[i][3]),
                    float(conf[i])
                ])
        if len(dets_for_bt) > 0:
            dets_for_bt = torch.tensor(dets_for_bt, dtype=torch.float32, device="cpu")
        else:
            dets_for_bt = torch.empty((0, 6), dtype=torch.float32, device="cpu")

        img_h, img_w = frame.shape[:2]
        img_info = (img_h, img_w)
        img_size = (in_h, in_w)
        online_targets = tracker.update(dets_for_bt, img_info, img_size)

        # YENİ: sadece render=='all' ise çiz
        if args.render == "all":
            draw_tracks(frame, online_targets, names)

        # --- JSON: bu frame'in kutularını ekle
        frame_tracks = []
        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            frame_tracks.append({
                "id": int(t.track_id),
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2)
            })
        tracks_out[str(frame_idx)] = frame_tracks

        # Writer
        if out_writer is None:
            h_out_v, w_out_v = frame.shape[:2]
            out_writer = cv2.VideoWriter(str(out_path), fourcc, cap_fps, (w_out_v, h_out_v))
            if not out_writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {out_path}")
        out_writer.write(frame)

        # Show (opsiyonel)
        disp_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win_name, disp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    # --- JSON'u yaz ---
    json_path = out_path.with_suffix(".tracks.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tracks_out, f, ensure_ascii=False)
        print(f"[tracks] wrote {json_path}")
    except Exception as e:
        print(f"[tracks] write failed: {e}")

if __name__ == "__main__":
    main()
