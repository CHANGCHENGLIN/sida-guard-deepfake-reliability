"""
SIDA-Guard main pipeline.

This script samples real/fake videos, generates controlled perturbations,
extracts whole-frame and fixed-face-ROI quality metrics, and computes a
quality-aware reliability score.

Default Kaggle usage:
    python src/sida_guard_pipeline.py

Local usage:
    python src/sida_guard_pipeline.py --input_dir /path/to/dataset --output_dir ./SIDA_Guard
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = [
    ("clean", "none", "none", {}),
    ("light_flicker_whole", "light_flicker", "whole_frame", {"amplitude": 0.35}),
    ("motion_blur_whole", "motion_blur", "whole_frame", {"kernel_size": 11}),
    ("ghosting_whole", "ghosting", "whole_frame", {"previous_frame_weight": 0.35}),
    ("mixed_whole", "mixed", "whole_frame", {"light_flicker_amplitude": 0.25, "motion_blur_kernel_size": 9, "previous_frame_weight": 0.25}),
    ("light_flicker_face", "light_flicker", "face", {"amplitude": 0.35}),
    ("motion_blur_face", "motion_blur", "face", {"kernel_size": 11}),
    ("light_flicker_background", "light_flicker", "background", {"amplitude": 0.35}),
    ("motion_blur_background", "motion_blur", "background", {"kernel_size": 11}),
]


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.raw_real = root / "data" / "raw" / "real"
        self.raw_fake = root / "data" / "raw" / "fake"
        self.processed = root / "data" / "processed"
        self.results = root / "results"
        self.figures = root / "figures"
        self.previews = self.figures / "perturbation_previews"
        self.logs = root / "logs"
        for d in [self.raw_real, self.raw_fake, self.processed, self.results, self.figures, self.previews, self.logs]:
            d.mkdir(parents=True, exist_ok=True)


def log(paths: Paths, msg: str) -> None:
    print(msg)
    with open(paths.logs / "pipeline_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def find_videos(input_dir: Path) -> List[Path]:
    exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    files: List[Path] = []
    for ext in exts:
        files.extend(input_dir.rglob(f"*{ext}"))
    return files


def split_real_fake(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    real_keys = ["youtube-real", "celeb-real", "original", "authentic", "real"]
    fake_keys = ["celeb-synthesis", "synthesis", "deepfake", "manipulated", "forged", "fake"]
    real, fake = [], []
    for p in files:
        s = str(p).lower()
        if any(k in s for k in fake_keys):
            fake.append(p)
        elif any(k in s for k in real_keys):
            real.append(p)
    return real, fake


def sample_videos(paths: Paths, input_dir: Path, num_real: int, num_fake: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    files = find_videos(input_dir)
    real, fake = split_real_fake(files)
    rows = []
    for label, candidates, n, outdir in [("real", real, num_real, paths.raw_real), ("fake", fake, num_fake, paths.raw_fake)]:
        n = min(n, len(candidates))
        for i, src in enumerate(random.sample(candidates, n), 1):
            vid = f"{label}_{i:03d}"
            dst = outdir / f"{vid}.mp4"
            shutil.copy2(src, dst)
            rows.append({"video_id": vid, "label": label, "new_path": str(dst), "original_path": str(src)})
    df = pd.DataFrame(rows)
    df.to_csv(paths.results / "sampled_videos_metadata.csv", index=False)
    return df


def load_face_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return None if detector.empty() else detector


FACE_DETECTOR = load_face_detector()


def detect_face(frame) -> Optional[Tuple[int, int, int, int]]:
    if FACE_DETECTOR is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    return tuple(map(int, max(faces, key=lambda b: b[2] * b[3])))


def expand_bbox(bbox, shape, scale=1.25):
    x, y, w, h = bbox
    H, W = shape[:2]
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * scale, h * scale
    return max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2)), min(W, int(cx + nw / 2)), min(H, int(cy + nh / 2))


def apply_light_flicker(frame, idx, amplitude=0.35):
    factor = 1.0 + amplitude * np.sin(idx * 0.35)
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_motion_blur(frame, kernel_size=11):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(frame, -1, kernel)


def apply_ghosting(frame, prev, weight=0.35):
    if prev is None:
        return frame
    return cv2.addWeighted(frame, 1.0 - weight, prev, weight, 0)


def perturb_full(frame, prev, idx, ptype, params):
    if ptype == "none":
        return frame.copy()
    if ptype == "light_flicker":
        return apply_light_flicker(frame, idx, params.get("amplitude", 0.35))
    if ptype == "motion_blur":
        return apply_motion_blur(frame, params.get("kernel_size", 11))
    if ptype == "ghosting":
        return apply_ghosting(frame, prev, params.get("previous_frame_weight", 0.35))
    if ptype == "mixed":
        out = apply_light_flicker(frame, idx, params.get("light_flicker_amplitude", 0.25))
        out = apply_motion_blur(out, params.get("motion_blur_kernel_size", 9))
        return apply_ghosting(out, prev, params.get("previous_frame_weight", 0.25))
    return frame.copy()


def mean_abs_diff(a, b):
    return float(np.mean(cv2.absdiff(a, b)))


def save_preview(paths: Paths, original, perturbed, name: str):
    diff = cv2.absdiff(original, perturbed)
    heat = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    panel = np.concatenate([original, perturbed, heat], axis=1)
    cv2.imwrite(str(paths.previews / f"{name}.jpg"), panel)


def generate_perturbations(paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = [(p, "real") for p in sorted(paths.raw_real.glob("*.mp4"))] + [(p, "fake") for p in sorted(paths.raw_fake.glob("*.mp4"))]
    processed_rows, meta_rows = [], []
    for src, label in raw:
        vid = src.stem
        cap0 = cv2.VideoCapture(str(src))
        fps = cap0.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)); N = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
        cap0.release()
        start, end = int(N * 0.2), int(N * 0.8)
        for condition, ptype, region, params in CONDITIONS:
            out_dir = paths.processed / condition; out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{vid}__{condition}.mp4"
            cap = cv2.VideoCapture(str(src))
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            prev, idx, modified, affected, face_found, first_preview = None, 0, 0, 0, 0, False
            diffs = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                original = frame.copy(); out = frame.copy(); roi_box = None
                if ptype != "none" and start <= idx <= end:
                    affected += 1
                    full = perturb_full(frame, prev, idx, ptype, params)
                    if region == "whole_frame":
                        out = full
                    elif region in ["face", "background"]:
                        bbox = detect_face(frame)
                        if bbox is not None:
                            face_found += 1
                            roi_box = expand_bbox(bbox, frame.shape, 1.3)
                            x1,y1,x2,y2 = roi_box
                            if region == "face":
                                out = frame.copy(); out[y1:y2, x1:x2] = full[y1:y2, x1:x2]
                            else:
                                out = full.copy(); out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    d = mean_abs_diff(original, out); diffs.append(d)
                    if d > 0.5: modified += 1
                    if not first_preview:
                        save_preview(paths, original, out, f"{vid}__{condition}")
                        first_preview = True
                writer.write(out)
                prev = original.copy(); idx += 1
            cap.release(); writer.release()
            processed_rows.append({"video_id": vid, "label": label, "condition": condition, "output_path": str(out_path)})
            meta_rows.append({"video_id": vid, "label": label, "condition": condition, "perturbation_type": ptype, "region": region,
                              "params": json.dumps(params), "total_frames": N, "affected_frames": affected, "modified_frames": modified,
                              "face_found_rate": (face_found / affected if affected and region in ["face", "background"] else np.nan),
                              "avg_roi_difference_score": float(np.mean(diffs)) if diffs else 0.0})
    processed = pd.DataFrame(processed_rows); meta = pd.DataFrame(meta_rows)
    processed.to_csv(paths.results / "processed_videos_metadata.csv", index=False)
    meta.to_csv(paths.results / "perturbation_metadata.csv", index=False)
    return processed, meta


def resize_frame(frame, width=320):
    h, w = frame.shape[:2]
    if w <= width: return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def quality_metrics_for_video(path: Path, frame_step=5, width=320):
    cap = cv2.VideoCapture(str(path))
    brightness, blur, temporal = [], [], []
    prev = None; idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % frame_step == 0:
            frame = resize_frame(frame, width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.mean(gray)))
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if prev is not None:
                temporal.append(mean_abs_diff(gray, prev))
            prev = gray
        idx += 1
    cap.release()
    if not brightness: return None
    b = np.array(brightness); bl = np.array(blur); t = np.array(temporal) if temporal else np.array([0.0])
    return {"brightness_mean": float(np.mean(b)), "brightness_std": float(np.std(b)), "brightness_range": float(np.max(b)-np.min(b)),
            "flicker_score": float(np.std(b)), "blur_score_mean": float(np.mean(bl)), "blur_score_min": float(np.min(bl)),
            "temporal_instability_mean": float(np.mean(t)), "temporal_instability_max": float(np.max(t))}


def compute_quality(paths: Paths, processed: pd.DataFrame, meta: pd.DataFrame):
    rows = []
    for _, r in processed.iterrows():
        m = quality_metrics_for_video(Path(r.output_path))
        if m:
            row = r.to_dict(); row.update(m); rows.append(row)
    df = pd.DataFrame(rows).merge(meta, on=["video_id", "label", "condition"], how="left")
    df.to_csv(paths.results / "quality_metrics.csv", index=False)
    summary = df.groupby("condition").mean(numeric_only=True).reset_index()
    summary.to_csv(paths.results / "quality_metrics_summary_by_condition.csv", index=False)
    return df, summary


def compute_fixed_face_roi(paths: Paths, processed: pd.DataFrame, quality: pd.DataFrame, frame_step=5, width=320, roi_size=128):
    tracks = []
    for _, r in processed[processed.condition == "clean"].iterrows():
        cap = cv2.VideoCapture(str(r.output_path)); idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % frame_step == 0:
                small = resize_frame(frame, width); bbox = detect_face(small)
                if bbox is None: tracks.append({"video_id": r.video_id, "frame_idx": idx, "face_found": False})
                else:
                    x1,y1,x2,y2 = expand_bbox(bbox, small.shape, 1.25)
                    tracks.append({"video_id": r.video_id, "frame_idx": idx, "face_found": True, "x1":x1,"y1":y1,"x2":x2,"y2":y2})
            idx += 1
        cap.release()
    track_df = pd.DataFrame(tracks); track_df.to_csv(paths.results / "fixed_face_tracks.csv", index=False)
    rows = []
    for _, r in processed.iterrows():
        tv = track_df[track_df.video_id == r.video_id]
        track = {int(x.frame_idx): x for _, x in tv.iterrows()}
        cap = cv2.VideoCapture(str(r.output_path)); idx = 0; vals=[]; blurs=[]; temps=[]; prev=None; analyzed=0; available=0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % frame_step == 0:
                analyzed += 1; small = resize_frame(frame, width)
                if idx in track and bool(track[idx].face_found):
                    tr = track[idx]; x1,y1,x2,y2 = int(tr.x1), int(tr.y1), int(tr.x2), int(tr.y2)
                    roi = small[y1:y2, x1:x2]
                    if roi.size:
                        roi = cv2.resize(roi, (roi_size, roi_size)); gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        vals.append(float(np.mean(gray))); blurs.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                        if prev is not None: temps.append(mean_abs_diff(gray, prev))
                        prev = gray; available += 1
            idx += 1
        cap.release()
        vals = np.array(vals) if vals else np.array([0.0]); blurs = np.array(blurs) if blurs else np.array([0.0]); temps = np.array(temps) if temps else np.array([0.0])
        rows.append({"video_id": r.video_id, "label": r.label, "condition": r.condition, "fixed_face_detection_rate": available/max(analyzed,1),
                     "fixed_face_flicker_score": float(np.std(vals)), "fixed_face_blur_score_mean": float(np.mean(blurs)), "fixed_face_blur_score_min": float(np.min(blurs)),
                     "fixed_face_temporal_instability_mean": float(np.mean(temps))})
    face = pd.DataFrame(rows); face.to_csv(paths.results / "fixed_face_roi_quality_metrics.csv", index=False)
    merged = quality.merge(face, on=["video_id","label","condition"], how="left")
    merged.to_csv(paths.results / "quality_metrics_with_fixed_face_roi.csv", index=False)
    summary = merged.groupby("condition").mean(numeric_only=True).reset_index()
    summary.to_csv(paths.results / "fixed_face_roi_quality_summary_by_condition.csv", index=False)
    return face, merged, summary


def risk_increase(cur, base, scale=3.0): return np.clip(((cur-base)/(base+1e-6))/scale,0,1)
def risk_decrease(cur, base, scale=0.5): return np.clip(((base-cur)/(base+1e-6))/scale,0,1)


def compute_risk(paths: Paths, df: pd.DataFrame):
    clean = df[df.condition == "clean"]
    base = clean[["video_id","flicker_score","blur_score_mean","temporal_instability_mean","fixed_face_flicker_score","fixed_face_blur_score_mean","fixed_face_temporal_instability_mean"]].copy()
    base.columns = ["video_id","b_flicker","b_blur","b_temp","b_face_flicker","b_face_blur","b_face_temp"]
    m = df.merge(base, on="video_id", how="left")
    m["global_flicker_risk"] = risk_increase(m.flicker_score, m.b_flicker)
    m["global_blur_risk"] = risk_decrease(m.blur_score_mean, m.b_blur)
    m["global_temporal_risk"] = risk_increase(m.temporal_instability_mean, m.b_temp)
    m["global_quality_risk"] = 0.45*m.global_flicker_risk + 0.35*m.global_blur_risk + 0.20*m.global_temporal_risk
    m["face_flicker_risk"] = risk_increase(m.fixed_face_flicker_score, m.b_face_flicker)
    m["face_blur_risk"] = risk_decrease(m.fixed_face_blur_score_mean, m.b_face_blur)
    m["face_temporal_risk"] = risk_increase(m.fixed_face_temporal_instability_mean, m.b_face_temp)
    m["face_quality_risk"] = 0.45*m.face_flicker_risk + 0.40*m.face_blur_risk + 0.15*m.face_temporal_risk
    m["final_quality_risk"] = 0.40*m.global_quality_risk + 0.60*m.face_quality_risk
    m["quality_reliability_score"] = 1.0 - m.final_quality_risk
    m["guard_decision"] = np.where(m.final_quality_risk >= 0.65, "abstain", np.where(m.final_quality_risk >= 0.35, "warning", "accept"))
    m.to_csv(paths.results / "sida_guard_quality_only.csv", index=False)
    summary = m.groupby("condition").mean(numeric_only=True).reset_index().sort_values("final_quality_risk", ascending=False)
    summary.to_csv(paths.results / "sida_guard_quality_summary_by_condition.csv", index=False)
    return m, summary


def plot_bar(paths: Paths, df: pd.DataFrame, col: str, fname: str, title: str, ascending=False):
    if col not in df.columns: return
    p = df.sort_values(col, ascending=ascending)
    plt.figure(figsize=(12,6)); plt.bar(p.condition, p[col]); plt.xticks(rotation=45, ha="right"); plt.title(title); plt.tight_layout(); plt.savefig(paths.figures / fname, dpi=200); plt.close()


def generate_figures(paths: Paths, quality_summary, face_summary, risk_summary):
    plot_bar(paths, quality_summary, "flicker_score", "whole_frame_flicker_score_by_condition.png", "Whole-frame Flicker Score")
    plot_bar(paths, quality_summary, "blur_score_mean", "whole_frame_blur_score_by_condition.png", "Whole-frame Blur Score", ascending=True)
    plot_bar(paths, face_summary, "fixed_face_flicker_score", "fixed_face_flicker_score_by_condition.png", "Fixed Face ROI Flicker Score")
    plot_bar(paths, face_summary, "fixed_face_blur_score_mean", "fixed_face_blur_score_by_condition.png", "Fixed Face ROI Blur Score", ascending=True)
    plot_bar(paths, risk_summary, "final_quality_risk", "sida_guard_quality_risk_by_condition.png", "SIDA-Guard Final Quality Risk")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/kaggle/input")
    parser.add_argument("--output_dir", default="/kaggle/working/SIDA_Guard")
    parser.add_argument("--num_real", type=int, default=3)
    parser.add_argument("--num_fake", type=int, default=3)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if args.clear and out.exists(): shutil.rmtree(out)
    paths = Paths(out)
    log(paths, "Starting SIDA-Guard pipeline")
    sampled = sample_videos(paths, Path(args.input_dir), args.num_real, args.num_fake)
    if sampled.empty:
        raise RuntimeError("No videos sampled. Check dataset path and real/fake keywords.")
    processed, meta = generate_perturbations(paths)
    quality, qsum = compute_quality(paths, processed, meta)
    face, merged, fsum = compute_fixed_face_roi(paths, processed, quality)
    risk, rsum = compute_risk(paths, merged)
    generate_figures(paths, qsum, fsum, rsum)
    log(paths, "Pipeline finished successfully")


if __name__ == "__main__":
    main()
