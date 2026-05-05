#!/usr/bin/env python3
"""
Lab 4 - Stereo camera calibration with OpenCV

This script is designed for a side-by-side stereo stream such as the ZED family.
It can:
1) capture stereo calibration pairs from a live camera
2) detect checkerboard corners on saved pairs
3) compute mono + stereo calibration
4) save calibration results to NPZ and JSON
5) preview the rectified stereo stream

Typical usage:
    python Lab4_Camera_Calibration_Local.py --mode all --camera-name zed2i_groupA \
        --device-id 0 --frame-width 2560 --frame-height 720 \
        --board-cols 9 --board-rows 6 --square-size-mm 24 --num-pairs 25

Keyboard shortcuts:
    q  quit current mode
    s  save a pair manually during capture (if corners are found on both views)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class BoardConfig:
    cols: int
    rows: int
    square_size_mm: float

    @property
    def size(self) -> Tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp[:, :2] = grid
        objp *= self.square_size_mm
        return objp


@dataclass
class StereoPair:
    left_path: Path
    right_path: Path


def ensure_dirs(base_dir: Path) -> None:
    for rel in [
        "left",
        "right",
        "corners/left",
        "corners/right",
        "results",
    ]:
        (base_dir / rel).mkdir(parents=True, exist_ok=True)


def draw_status(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    out = img.copy()
    y = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return out


def split_stereo_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    mid = w // 2
    left = frame[:, :mid].copy()
    right = frame[:, mid: mid * 2].copy()
    return left, right


def open_capture(device_id: int, frame_width: int | None, frame_height: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device_id}.")

    if frame_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
    if frame_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))

    return cap


def find_checkerboard(gray: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
        if found:
            return True, corners.astype(np.float32)

    flags_legacy = flags | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, board_size, flags_legacy)
    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def capture_pairs(
    base_dir: Path,
    board: BoardConfig,
    device_id: int,
    frame_width: int | None,
    frame_height: int | None,
    num_pairs: int,
    capture_interval: float,
    preview_scale: float,
) -> List[StereoPair]:
    ensure_dirs(base_dir)
    cap = open_capture(device_id, frame_width, frame_height)

    saved_pairs: List[StereoPair] = []
    last_capture_time = 0.0

    print("Live capture started.")
    print("Press 'q' to stop, 's' to save manually when corners are visible on both images.")

    try:
        while len(saved_pairs) < num_pairs:
            ok, frame = cap.read()
            if not ok:
                print("Warning: could not read a frame from the camera.")
                continue

            left, right = split_stereo_frame(frame)
            gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            found_left, corners_left = find_checkerboard(gray_left, board.size)
            found_right, corners_right = find_checkerboard(gray_right, board.size)

            left_vis = left.copy()
            right_vis = right.copy()

            if found_left and corners_left is not None:
                cv2.drawChessboardCorners(left_vis, board.size, corners_left, found_left)
            if found_right and corners_right is not None:
                cv2.drawChessboardCorners(right_vis, board.size, corners_right, found_right)

            lines = [
                f"Saved pairs: {len(saved_pairs)}/{num_pairs}",
                f"Board detected: L={found_left}  R={found_right}",
                "Move the board: center, corners, different tilts, different distances",
            ]
            left_vis = draw_status(left_vis, lines)
            right_vis = draw_status(right_vis, lines)

            if preview_scale != 1.0:
                left_vis = cv2.resize(left_vis, None, fx=preview_scale, fy=preview_scale)
                right_vis = cv2.resize(right_vis, None, fx=preview_scale, fy=preview_scale)

            cv2.imshow("Left preview", left_vis)
            cv2.imshow("Right preview", right_vis)

            now = time.time()
            auto_save = (
                found_left
                and found_right
                and (now - last_capture_time) >= capture_interval
            )

            key = cv2.waitKey(1) & 0xFF
            manual_save = key == ord("s") and found_left and found_right

            if auto_save or manual_save:
                pair_id = len(saved_pairs) + 1
                left_path = base_dir / "left" / f"pair_{pair_id:03d}_left.png"
                right_path = base_dir / "right" / f"pair_{pair_id:03d}_right.png"
                left_corner_path = base_dir / "corners/left" / f"pair_{pair_id:03d}_left_corners.png"
                right_corner_path = base_dir / "corners/right" / f"pair_{pair_id:03d}_right_corners.png"

                cv2.imwrite(str(left_path), left)
                cv2.imwrite(str(right_path), right)
                cv2.imwrite(str(left_corner_path), left_vis)
                cv2.imwrite(str(right_corner_path), right_vis)

                saved_pairs.append(StereoPair(left_path=left_path, right_path=right_path))
                last_capture_time = now
                print(f"Saved pair {pair_id:03d}")

            if key == ord("q"):
                print("Capture interrupted by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved_pairs


def load_pairs(base_dir: Path) -> List[StereoPair]:
    left_paths = sorted((base_dir / "left").glob("*.png")) + sorted((base_dir / "left").glob("*.jpg"))
    right_paths = sorted((base_dir / "right").glob("*.png")) + sorted((base_dir / "right").glob("*.jpg"))

    if not left_paths or not right_paths:
        raise FileNotFoundError(
            f"No stereo pairs found in {base_dir}. Expected images in 'left/' and 'right/'."
        )
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"Mismatched number of images: left={len(left_paths)}, right={len(right_paths)}."
        )

    return [StereoPair(left_path=l, right_path=r) for l, r in zip(left_paths, right_paths)]


def collect_calibration_points(
    pairs: Sequence[StereoPair],
    board: BoardConfig,
    corner_dir: Path | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int], List[StereoPair]]:
    object_points: List[np.ndarray] = []
    image_points_left: List[np.ndarray] = []
    image_points_right: List[np.ndarray] = []
    valid_pairs: List[StereoPair] = []
    image_size: Tuple[int, int] | None = None

    if corner_dir is not None:
        (corner_dir / "left").mkdir(parents=True, exist_ok=True)
        (corner_dir / "right").mkdir(parents=True, exist_ok=True)

    objp = board.object_points()

    for idx, pair in enumerate(pairs, start=1):
        left = cv2.imread(str(pair.left_path))
        right = cv2.imread(str(pair.right_path))
        if left is None or right is None:
            print(f"Skipping unreadable pair {idx}: {pair.left_path.name}, {pair.right_path.name}")
            continue

        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        image_size = (gray_left.shape[1], gray_left.shape[0])

        found_left, corners_left = find_checkerboard(gray_left, board.size)
        found_right, corners_right = find_checkerboard(gray_right, board.size)

        if found_left and found_right and corners_left is not None and corners_right is not None:
            object_points.append(objp.copy())
            image_points_left.append(corners_left)
            image_points_right.append(corners_right)
            valid_pairs.append(pair)

            if corner_dir is not None:
                vis_left = left.copy()
                vis_right = right.copy()
                cv2.drawChessboardCorners(vis_left, board.size, corners_left, True)
                cv2.drawChessboardCorners(vis_right, board.size, corners_right, True)
                cv2.imwrite(str(corner_dir / "left" / f"pair_{idx:03d}_corners.png"), vis_left)
                cv2.imwrite(str(corner_dir / "right" / f"pair_{idx:03d}_corners.png"), vis_right)
        else:
            print(f"Checkerboard not found in both views for pair {idx:03d}")

    if image_size is None:
        raise RuntimeError("No readable image pairs found.")
    if len(valid_pairs) < 8:
        raise RuntimeError(
            f"Only {len(valid_pairs)} valid pairs available. Please capture at least 8-12 good stereo pairs."
        )

    return object_points, image_points_left, image_points_right, image_size, valid_pairs


def reprojection_error(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[float, List[float]]:
    per_view_errors: List[float] = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))
    return float(np.mean(per_view_errors)), per_view_errors


def calibrate_from_pairs(base_dir: Path, board: BoardConfig) -> dict:
    ensure_dirs(base_dir)
    pairs = load_pairs(base_dir)
    objpoints, imgpoints_left, imgpoints_right, image_size, valid_pairs = collect_calibration_points(
        pairs,
        board,
        corner_dir=base_dir / "corners",
    )

    print(f"Total pairs found: {len(pairs)}")
    print(f"Valid pairs used for calibration: {len(valid_pairs)}")
    print(f"Image size (one view): {image_size[0]} x {image_size[1]}")

    rms_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints,
        imgpoints_left,
        image_size,
        None,
        None,
    )

    rms_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints,
        imgpoints_right,
        image_size,
        None,
        None,
    )

    mean_err_left, per_view_left = reprojection_error(
        objpoints, imgpoints_left, rvecs_left, tvecs_left, K_left, dist_left
    )
    mean_err_right, per_view_right = reprojection_error(
        objpoints, imgpoints_right, rvecs_right, tvecs_right, K_right, dist_right
    )

    stereo_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    stereo_rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_left,
        dist_left,
        K_right,
        dist_right,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left,
        dist_left,
        K_right,
        dist_right,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K_left,
        dist_left,
        R1,
        P1,
        image_size,
        cv2.CV_32FC1,
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K_right,
        dist_right,
        R2,
        P2,
        image_size,
        cv2.CV_32FC1,
    )

    results = {
        "image_size": np.array(image_size, dtype=np.int32),
        "board_size": np.array([board.cols, board.rows], dtype=np.int32),
        "square_size_mm": np.array([board.square_size_mm], dtype=np.float32),
        "K_left": K_left,
        "dist_left": dist_left,
        "K_right": K_right,
        "dist_right": dist_right,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi1": np.array(roi1, dtype=np.int32),
        "roi2": np.array(roi2, dtype=np.int32),
        "map1_left": map1_left,
        "map2_left": map2_left,
        "map1_right": map1_right,
        "map2_right": map2_right,
        "rms_left": np.array([rms_left], dtype=np.float32),
        "rms_right": np.array([rms_right], dtype=np.float32),
        "stereo_rms": np.array([stereo_rms], dtype=np.float32),
        "mean_reprojection_error_left": np.array([mean_err_left], dtype=np.float32),
        "mean_reprojection_error_right": np.array([mean_err_right], dtype=np.float32),
        "per_view_errors_left": np.array(per_view_left, dtype=np.float32),
        "per_view_errors_right": np.array(per_view_right, dtype=np.float32),
    }

    npz_path = base_dir / "results" / "stereo_calibration.npz"
    np.savez_compressed(npz_path, **results)

    summary = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "board_size": [board.cols, board.rows],
        "square_size_mm": board.square_size_mm,
        "num_total_pairs": len(pairs),
        "num_valid_pairs": len(valid_pairs),
        "rms_left": float(rms_left),
        "rms_right": float(rms_right),
        "stereo_rms": float(stereo_rms),
        "mean_reprojection_error_left": float(mean_err_left),
        "mean_reprojection_error_right": float(mean_err_right),
        "baseline_mm": float(np.linalg.norm(T)),
    }

    summary_path = base_dir / "results" / "stereo_calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCalibration completed.")
    print(f"Left RMS:  {rms_left:.4f}")
    print(f"Right RMS: {rms_right:.4f}")
    print(f"Stereo RMS: {stereo_rms:.4f}")
    print(f"Mean reprojection error (left):  {mean_err_left:.4f} px")
    print(f"Mean reprojection error (right): {mean_err_right:.4f} px")
    print(f"Estimated stereo baseline: {np.linalg.norm(T):.2f} mm")
    print(f"Saved NPZ: {npz_path}")
    print(f"Saved summary: {summary_path}")

    return results


def load_results(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def preview_rectification(
    results: dict,
    device_id: int,
    frame_width: int | None,
    frame_height: int | None,
    preview_scale: float,
) -> None:
    cap = open_capture(device_id, frame_width, frame_height)
    map1_left = results["map1_left"]
    map2_left = results["map2_left"]
    map1_right = results["map1_right"]
    map2_right = results["map2_right"]

    print("Rectification preview started. Press 'q' to exit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: could not read a frame from the camera.")
                continue

            left, right = split_stereo_frame(frame)
            rect_left = cv2.remap(left, map1_left, map2_left, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right, map1_right, map2_right, cv2.INTER_LINEAR)

            combined = np.hstack([rect_left, rect_right])
            h, w = combined.shape[:2]
            for y in range(40, h, 40):
                cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)

            combined = draw_status(
                combined,
                [
                    "Rectified stereo preview",
                    "Horizontal lines should pass through corresponding points",
                    "Press q to close",
                ],
            )

            if preview_scale != 1.0:
                combined = cv2.resize(combined, None, fx=preview_scale, fy=preview_scale)

            cv2.imshow("Rectified stereo preview", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo camera calibration lab script.")
    parser.add_argument("--mode", choices=["capture", "calibrate", "preview", "all"], default="all")
    parser.add_argument("--camera-name", type=str, default="zed_group")
    parser.add_argument("--output-root", type=str, default="lab4_data")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument("--board-cols", type=int, default=9, help="Number of inner corners along columns.")
    parser.add_argument("--board-rows", type=int, default=6, help="Number of inner corners along rows.")
    parser.add_argument("--square-size-mm", type=float, default=24.0)
    parser.add_argument("--num-pairs", type=int, default=25)
    parser.add_argument("--capture-interval", type=float, default=1.5)
    parser.add_argument("--preview-scale", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board = BoardConfig(
        cols=args.board_cols,
        rows=args.board_rows,
        square_size_mm=args.square_size_mm,
    )
    base_dir = Path(args.output_root) / args.camera_name
    ensure_dirs(base_dir)

    if args.mode in {"capture", "all"}:
        capture_pairs(
            base_dir=base_dir,
            board=board,
            device_id=args.device_id,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            num_pairs=args.num_pairs,
            capture_interval=args.capture_interval,
            preview_scale=args.preview_scale,
        )

    results = None
    if args.mode in {"calibrate", "all"}:
        results = calibrate_from_pairs(base_dir, board)

    if args.mode == "preview":
        npz_path = base_dir / "results" / "stereo_calibration.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {npz_path}")
        results = load_results(npz_path)

    if args.mode in {"preview", "all"}:
        assert results is not None
        preview_rectification(
            results=results,
            device_id=args.device_id,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            preview_scale=args.preview_scale,
        )


if __name__ == "__main__":
    main()
