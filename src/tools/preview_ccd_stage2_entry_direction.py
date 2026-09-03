from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import CCD_STAGE2_BOTSORT_TRACKS, CCD_STAGE2_MANIFEST, STAGE2_MANIFEST
from src.tools.build_ccd_stage2_entry_direction import ego_lane_proxy_polygon, select_target_track

MANIFEST_PATH = CCD_STAGE2_MANIFEST / "ego_candidates.csv"
PSEUDO_LABEL_PATH = STAGE2_MANIFEST / "ccd_stage2_entry_direction_pseudo_labels.csv"
TRACK_DIR = CCD_STAGE2_BOTSORT_TRACKS
OUTPUT_DIR = STAGE2_MANIFEST / "entry_direction_preview"


def normalize_video_id(value: object) -> str:
    return str(value).zfill(6)


def load_row(video_id: str, pseudo_label_path: Path = PSEUDO_LABEL_PATH) -> tuple[pd.Series, pd.Series]:
    video_id = normalize_video_id(video_id)
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"video_id": str})
    manifest["video_id"] = manifest["video_id"].map(normalize_video_id)
    pseudo = pd.read_csv(pseudo_label_path, dtype={"video_id": str})
    pseudo["video_id"] = pseudo["video_id"].map(normalize_video_id)
    m = manifest[manifest["video_id"] == video_id]
    p = pseudo[pseudo["video_id"] == video_id]
    if m.empty or p.empty:
        raise RuntimeError(f"video_id not found in manifest/pseudo labels: {video_id}")
    return m.iloc[0], p.iloc[0]


def load_tracks(video_id: str) -> pd.DataFrame:
    path = TRACK_DIR / f"{normalize_video_id(video_id)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["video_id"] = df["video_id"].map(normalize_video_id)
    return df


def draw_x_trace(frame: np.ndarray, track: pd.DataFrame, label: pd.Series) -> None:
    if track.empty:
        return
    h, w = frame.shape[:2]
    xs = track.sort_values("frame")["bottom_x_norm"].astype(float).to_numpy()
    if len(xs) < 2:
        return
    box_x, box_y, box_w, box_h = 15, h - 105, 220, 70
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    pts = []
    for i, x in enumerate(xs):
        px = box_x + int(i / max(1, len(xs) - 1) * box_w)
        py = box_y + box_h - int(max(0.0, min(1.0, x)) * box_h)
        pts.append((px, py))
    for p1, p2 in zip(pts, pts[1:]):
        cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
    start_x, mid_x, end_x = float(xs[0]), float(xs[len(xs) // 2]), float(xs[-1])
    dx = end_x - start_x
    cv2.putText(frame, f"x s/m/e={start_x:.2f}/{mid_x:.2f}/{end_x:.2f} dx={dx:.2f}", (box_x, box_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"lat={label.get('direction_lateral_displacement', 0):.2f}", (box_x + 120, box_y + box_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, tracks: pd.DataFrame, label: pd.Series, frame_idx: int, candidate_by_track: dict[int, dict] | None = None) -> None:
    h, w = frame.shape[:2]
    cv2.polylines(frame, [ego_lane_proxy_polygon(w, h)], True, (255, 255, 0), 2)
    candidate_by_track = candidate_by_track or {}
    target_id = None if pd.isna(label["target_track_id"]) else int(label["target_track_id"])
    for tid, track in tracks.groupby("track_id"):
        pts = track.sort_values("frame")
        points = [(int(r["bottom_x"]), int(r["bottom_y"])) for _, r in pts.iterrows()]
        color = (0, 0, 255) if int(tid) == target_id else (0, 180, 0)
        for p1, p2 in zip(points, points[1:]):
            cv2.line(frame, p1, p2, color, 2 if int(tid) == target_id else 1, cv2.LINE_AA)
    current = tracks[tracks["frame"].astype(int) == frame_idx]
    for _, row in current.iterrows():
        tid = int(row["track_id"])
        target = tid == target_id
        color = (0, 0, 255) if target else (0, 180, 0)
        thickness = 3 if target else 1
        x1, y1, x2, y2 = [int(row[k]) for k in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (int(row["bottom_x"]), int(row["bottom_y"])), 4, color, -1)
        cand = candidate_by_track.get(tid, {})
        label_text = f"#{tid}"
        if cand:
            label_text += f" r{int(cand.get('candidate_rank', 0))} s{cand.get('target_score', 0):.2f} l{int(cand.get('track_length', 0))} p{cand.get('collision_proximity_score', 0):.2f}"
        else:
            label_text += f" l{int(tracks[tracks['track_id'] == tid]['frame'].nunique())}"
        cv2.putText(frame, label_text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    entry = None if pd.isna(label["entry_frame"]) else int(label["entry_frame"])
    collision = int(label["collision_frame"])
    if frame_idx == entry:
        cv2.rectangle(frame, (4, 4), (w - 5, h - 5), (255, 0, 255), 5)
        cv2.putText(frame, "ENTRY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3, cv2.LINE_AA)
    if frame_idx == collision:
        cv2.rectangle(frame, (10, 10), (w - 11, h - 11), (0, 0, 255), 5)
        cv2.putText(frame, "COLLISION", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    text = f"frame={frame_idx} status={label.get('stage2_entry_status', '')} target={target_id} target_score={label['target_score']:.2f} collision={collision} end={label.get('track_end_frame', '')}"
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    top3 = sorted(candidate_by_track.values(), key=lambda c: c.get("candidate_rank", 999))[:3]
    for i, cand in enumerate(top3):
        line = f"top{i + 1}: #{int(cand['track_id'])} s={cand.get('target_score', 0):.2f} prox={cand.get('collision_proximity_score', 0):.2f} end={cand.get('track_end_frame', '')} {cand.get('target_reject_reason', '')}"
        cv2.putText(frame, line, (15, 52 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if target_id is not None:
        draw_x_trace(frame, tracks[tracks["track_id"].astype(int) == target_id], label)


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def write_preview_video(video_id: str, video_path: Path, tracks: pd.DataFrame, label: pd.Series, output_dir: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = output_dir / f"{video_id}_entry_direction.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    _, candidates, _ = select_target_track(tracks, int(label["collision_frame"]))
    candidate_by_track = {int(c["track_id"]): c for c in candidates}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_overlay(frame, tracks, label, frame_idx, candidate_by_track)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return output


def write_contact_sheet(video_id: str, video_path: Path, tracks: pd.DataFrame, label: pd.Series, output_dir: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    collision = int(label["collision_frame"])
    frames = [collision - 20, collision - 10, collision - 5, collision, collision + 2]
    _, candidates, _ = select_target_track(tracks, collision)
    candidate_by_track = {int(c["track_id"]): c for c in candidates}
    images = []
    for frame_idx in frames:
        frame_idx = max(0, min(total - 1, int(frame_idx)))
        frame = read_frame(cap, frame_idx)
        if frame is None:
            continue
        draw_overlay(frame, tracks, label, frame_idx, candidate_by_track)
        cv2.putText(frame, f"sample_frame={frame_idx}", (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        images.append(cv2.resize(frame, (320, 180)))
    cap.release()
    if not images:
        raise RuntimeError("No frames for contact sheet")
    sheet = np.vstack([np.hstack(images[:3]), np.hstack(images[3:] + [np.zeros_like(images[0])] * (3 - len(images[3:])))])
    output = output_dir / f"{video_id}_contact_sheet.jpg"
    cv2.imwrite(str(output), sheet)
    return output


def write_group_contact_sheets(labels: pd.DataFrame, pseudo_label_path: Path, output_dir: Path, per_group: int = 5) -> dict[str, int]:
    counts = {}
    for status, folder in (("valid", "valid"), ("not_observable", "not_observable"), ("ambiguous", "ambiguous"), ("no_target", "no_target")):
        group = labels[labels["stage2_entry_status"] == status].head(per_group)
        group_dir = output_dir / folder
        group_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        for video_id in group["video_id"].tolist():
            try:
                manifest, label = load_row(video_id, pseudo_label_path)
                tracks = load_tracks(video_id)
                write_contact_sheet(normalize_video_id(video_id), Path(manifest["video_path"]), tracks, label, group_dir)
                made += 1
            except Exception as exc:
                print(f"preview failed {video_id}: {exc}")
        counts[folder] = made
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview CCD Stage2 entry/direction pseudo-labels.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pseudo-label", type=Path, default=PSEUDO_LABEL_PATH)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    video_id = normalize_video_id(args.video_id)
    manifest, label = load_row(video_id, args.pseudo_label)
    tracks = load_tracks(video_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(manifest["video_path"])
    sheet = write_contact_sheet(video_id, video_path, tracks, label, args.output_dir)
    print(f"Saved: {sheet}")
    if not args.no_video:
        video = write_preview_video(video_id, video_path, tracks, label, args.output_dir)
        print(f"Saved: {video}")


if __name__ == "__main__":
    main()
