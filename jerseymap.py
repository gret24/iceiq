#!/usr/bin/env python3
"""
영상별 선수 히트맵 생성기
- 각 선수의 빙판 위 위치를 수집해 히트맵으로 시각화
- 영상마다 별도 저지맵 생성
- 출력: jerseymap_{영상명}_{선수번호}.png

사용법:
  python3 jerseymap.py <video_stem> [번호 번호 ...] [--all] [--out-dir /workspace]
  python3 jerseymap.py input_video 4 14 94
  python3 jerseymap.py input_video --all
  python3 jerseymap.py input_video --all --team HOME
"""

import argparse
import json
import os
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm._load_fontmanager(try_read_cache=False)
_nanum = [f.fname for f in _fm.fontManager.ttflist if 'NanumGothic' in f.name]
if _nanum:
    matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

# ── 경로 설정 ────────────────────────────────────────────
BASE_DIR    = "/root/iceiq"
FRAMES_DIR  = os.path.join(BASE_DIR, "frames")
OUT_DIR_DEFAULT = "/workspace"

# 하키 링크 실제 크기 (m) - 비율용
RINK_W = 60.0   # 폭 (가로)
RINK_H = 26.0   # 높이 (세로)

# 팀별 컬러맵
TEAM_CMAPS = {
    "HOME": "Reds",
    "AWAY": "Blues",
    "UNKNOWN": "Greens",
}
# 선수 개별 컬러 (여러 명 오버레이용)
PLAYER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#8bc34a",
]


def load_data(video_stem: str):
    """영상별 tracks.json / jersey_map.json 로드
    영상별 디렉토리가 있으면 그쪽, 없으면 기본 경로 사용"""
    video_dir = os.path.join(BASE_DIR, "videos", video_stem)
    if os.path.isdir(video_dir):
        tracks_path = os.path.join(video_dir, "tracks.json")
        jersey_path = os.path.join(video_dir, "jersey_map.json")
        frames_dir  = os.path.join(video_dir, "frames")
    else:
        # 기존 단일 경로 호환
        tracks_path = os.path.join(BASE_DIR, "tracks.json")
        jersey_path = os.path.join(BASE_DIR, "jersey_map.json")
        frames_dir  = FRAMES_DIR

    if not os.path.exists(tracks_path):
        print(f"❌ tracks.json 없음: {tracks_path}")
        sys.exit(1)
    if not os.path.exists(jersey_path):
        print(f"❌ jersey_map.json 없음: {jersey_path}")
        sys.exit(1)

    with open(tracks_path) as f:
        tracks = json.load(f)
    with open(jersey_path) as f:
        jersey_map = json.load(f)

    # 프레임 크기 감지
    frame_w, frame_h = 1280, 720
    for fname in sorted(tracks.keys()):
        if tracks[fname]:
            fpath = os.path.join(frames_dir, fname)
            if os.path.exists(fpath):
                img = cv2.imread(fpath)
                if img is not None:
                    frame_h, frame_w = img.shape[:2]
            break

    return tracks, jersey_map, frame_w, frame_h


def collect_positions(tracks: dict, jersey_map: dict,
                      target_numbers: list[str],
                      team_filter: str | None = None) -> dict:
    """
    선수별 위치 수집
    반환: {번호: {"positions": [(cx, cy), ...], "team": str}}
    """
    # track_id → (jersey번호, team) 매핑
    tid_to_info: dict[str, tuple[str, str]] = {}
    for tid_str, info in jersey_map.items():
        num = info["jersey"].lstrip("0") or "0"
        team = info.get("team", "UNKNOWN")
        if target_numbers and num not in target_numbers:
            continue
        if team_filter and team.upper() != team_filter.upper():
            continue
        tid_to_info[tid_str] = (num, team)

    if not tid_to_info:
        print(f"⚠ 해당 조건의 선수 없음 (번호={target_numbers}, 팀={team_filter})")
        return {}

    # 위치 수집
    result: dict[str, dict] = {}
    for num in (target_numbers if target_numbers else set(v[0] for v in tid_to_info.values())):
        result[num] = {"positions": [], "team": "UNKNOWN"}

    for fname in sorted(tracks.keys()):
        for t in tracks[fname]:
            tid = str(t["track_id"])
            if tid not in tid_to_info:
                continue
            num, team = tid_to_info[tid]
            cx = (t["x1"] + t["x2"]) / 2
            cy = (t["y1"] + t["y2"]) / 2
            result[num]["positions"].append((cx, cy))
            result[num]["team"] = team

    return result


def make_heatmap_single(positions: list[tuple], frame_w: int, frame_h: int,
                        sigma: int = 20) -> np.ndarray:
    """위치 리스트 → 정규화된 히트맵 2D array"""
    hmap = np.zeros((frame_h, frame_w), dtype=np.float32)
    for (cx, cy) in positions:
        x, y = int(cx), int(cy)
        if 0 <= x < frame_w and 0 <= y < frame_h:
            hmap[y, x] += 1.0
    hmap = gaussian_filter(hmap, sigma=sigma)
    if hmap.max() > 0:
        hmap /= hmap.max()
    return hmap


def draw_rink_overlay(ax, frame_w: int, frame_h: int):
    """빙판 라인 오버레이 (추정 기준, 영상 크기에 비례)"""
    # 링크 경계 (영상의 대략적인 빙판 영역 - 상하 10% 제외)
    pad_y_top    = int(frame_h * 0.08)
    pad_y_bottom = int(frame_h * 0.05)
    pad_x        = int(frame_w * 0.03)

    rink_x1 = pad_x
    rink_x2 = frame_w - pad_x
    rink_y1 = pad_y_top
    rink_y2 = frame_h - pad_y_bottom
    rw = rink_x2 - rink_x1
    rh = rink_y2 - rink_y1

    # 외곽 링크 경계
    rect = plt.Rectangle((rink_x1, rink_y1), rw, rh,
                          linewidth=2, edgecolor="white", facecolor="none", alpha=0.6)
    ax.add_patch(rect)

    # 센터 라인 (빨강)
    cx = frame_w // 2
    ax.axvline(cx, color="#ff4444", linewidth=1.5, alpha=0.7, linestyle="-")

    # 블루 라인 (양쪽)
    bl_left  = rink_x1 + int(rw * 0.25)
    bl_right = rink_x1 + int(rw * 0.75)
    ax.axvline(bl_left,  color="#4488ff", linewidth=1.2, alpha=0.6)
    ax.axvline(bl_right, color="#4488ff", linewidth=1.2, alpha=0.6)

    # 골 라인 (양쪽)
    gl_left  = rink_x1 + int(rw * 0.07)
    gl_right = rink_x1 + int(rw * 0.93)
    ax.axvline(gl_left,  color="#ff8888", linewidth=1.0, alpha=0.5)
    ax.axvline(gl_right, color="#ff8888", linewidth=1.0, alpha=0.5)

    # 페이스오프 서클 (센터)
    circle_r = int(rh * 0.28)
    center_circle = plt.Circle((cx, (rink_y1 + rink_y2) // 2), circle_r,
                                color="white", fill=False, linewidth=1.2, alpha=0.5)
    ax.add_patch(center_circle)


def save_jerseymap_single(num: str, info: dict, frame_w: int, frame_h: int,
                          video_stem: str, out_dir: str, color_idx: int = 0):
    """선수 1명 히트맵 저장"""
    positions = info["positions"]
    team      = info["team"]

    if not positions:
        print(f"  ⚠ {num}번: 위치 데이터 없음, 스킵")
        return None

    hmap = make_heatmap_single(positions, frame_w, frame_h)

    cmap_name = TEAM_CMAPS.get(team.upper(), "Oranges")
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0a1628")

    # 빙판 베이스 (연한 파랑-흰색)
    ice_base = np.ones((frame_h, frame_w, 4), dtype=np.float32)
    ice_base[:, :, 0] = 0.85   # R
    ice_base[:, :, 1] = 0.92   # G
    ice_base[:, :, 2] = 1.0    # B
    ice_base[:, :, 3] = 0.15   # alpha (흐릿하게)
    ax.imshow(ice_base, extent=[0, frame_w, frame_h, 0], aspect="auto")

    # 히트맵
    ax.imshow(hmap, cmap=cmap, alpha=0.85, vmin=0, vmax=1,
              extent=[0, frame_w, frame_h, 0], aspect="auto", interpolation="bilinear")

    # 실제 위치 점 (투명하게)
    if len(positions) <= 2000:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.scatter(xs, ys, c=PLAYER_COLORS[color_idx % len(PLAYER_COLORS)],
                   s=3, alpha=0.25, linewidths=0)

    # 링크 라인 오버레이
    draw_rink_overlay(ax, frame_w, frame_h)

    # 타이틀 & 레이블
    team_label = {"HOME": "🏠 HOME", "AWAY": "✈️ AWAY"}.get(team.upper(), team)
    ax.set_title(
        f"#{num}  |  {team_label}  |  {video_stem}\n"
        f"출전 감지: {len(positions)}회",
        color="white", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)
    ax.axis("off")

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("출전 밀도", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"jerseymap_{video_stem}_{num}.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {num}번 ({team}) → {out_path}  ({len(positions)}포인트)")
    return out_path


def save_jerseymap_combined(player_data: dict, frame_w: int, frame_h: int,
                             video_stem: str, out_dir: str, numbers: list[str]):
    """여러 선수 오버레이 합성 히트맵"""
    if len(numbers) < 2:
        return

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0a1628")

    # 빙판 베이스
    ice_base = np.ones((frame_h, frame_w, 4), dtype=np.float32)
    ice_base[:, :, :3] = [0.85, 0.92, 1.0]
    ice_base[:, :, 3] = 0.12
    ax.imshow(ice_base, extent=[0, frame_w, frame_h, 0], aspect="auto")

    legend_patches = []
    for i, num in enumerate(numbers):
        info = player_data.get(num, {})
        positions = info.get("positions", [])
        if not positions:
            continue
        hmap = make_heatmap_single(positions, frame_w, frame_h, sigma=25)

        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        # 단색 컬러맵 생성
        rgba = mcolors.to_rgba(color)
        cmap_custom = LinearSegmentedColormap.from_list(
            f"c{i}", [(rgba[0], rgba[1], rgba[2], 0), rgba], N=256
        )
        ax.imshow(hmap, cmap=cmap_custom, alpha=0.65, vmin=0, vmax=1,
                  extent=[0, frame_w, frame_h, 0], aspect="auto", interpolation="bilinear")

        team = info.get("team", "?")
        legend_patches.append(
            mpatches.Patch(color=color, label=f"#{num} ({team}) {len(positions)}회")
        )

    draw_rink_overlay(ax, frame_w, frame_h)

    nums_str = "_".join(numbers)
    ax.set_title(
        f"선수 비교 히트맵  |  {video_stem}\n"
        + "  ".join(f"#{n}" for n in numbers),
        color="white", fontsize=14, fontweight="bold", pad=12
    )
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=10, facecolor="#1a2a3a", labelcolor="white",
              framealpha=0.8, edgecolor="white")
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)
    ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"jerseymap_{video_stem}_combined_{nums_str}.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ 합성 히트맵 → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="영상별 선수 히트맵 생성기")
    parser.add_argument("video_stem", help="영상 이름 (확장자 제외, 예: input_video)")
    parser.add_argument("numbers", nargs="*", help="선수 번호 목록 (생략 시 --all 필요)")
    parser.add_argument("--all", action="store_true", help="jersey_map의 모든 선수")
    parser.add_argument("--team", default=None, help="팀 필터: HOME / AWAY")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--no-combined", action="store_true", help="합성 히트맵 생성 안함")
    parser.add_argument("--sigma", type=int, default=20, help="가우시안 블러 강도 (기본 20)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  영상별 선수 히트맵 생성기")
    print(f"  영상: {args.video_stem}  팀필터: {args.team or '전체'}")
    print(f"{'='*55}\n")

    tracks, jersey_map, frame_w, frame_h = load_data(args.video_stem)
    print(f"  프레임: {len(tracks)}개  크기: {frame_w}×{frame_h}")
    print(f"  jersey_map: {len(jersey_map)}개 Track\n")

    # 대상 번호 결정
    if args.all:
        all_nums = set()
        for info in jersey_map.values():
            num = info["jersey"].lstrip("0") or "0"
            if args.team is None or info.get("team","").upper() == args.team.upper():
                all_nums.add(num)
        target_numbers = sorted(all_nums, key=lambda x: int(x) if x.isdigit() else 999)
        print(f"  대상 선수: 전체 {len(target_numbers)}명 → {target_numbers}")
    else:
        if not args.numbers:
            print("❌ 번호를 지정하거나 --all 옵션을 사용하세요")
            sys.exit(1)
        target_numbers = [n.lstrip("0") or "0" for n in args.numbers]
        print(f"  대상 선수: {target_numbers}")

    print()

    # 위치 수집
    player_data = collect_positions(tracks, jersey_map, target_numbers, args.team)

    # 개별 히트맵 생성
    saved = []
    for i, num in enumerate(target_numbers):
        info = player_data.get(num)
        if not info:
            print(f"  ⚠ {num}번: 데이터 없음")
            continue
        out = save_jerseymap_single(num, info, frame_w, frame_h,
                                    args.video_stem, args.out_dir, color_idx=i)
        if out:
            saved.append(out)

    # 합성 히트맵 (2명 이상이고 --no-combined 아닐 때)
    if not args.no_combined and len(target_numbers) >= 2:
        print()
        save_jerseymap_combined(player_data, frame_w, frame_h,
                                args.video_stem, args.out_dir, target_numbers)

    print(f"\n{'='*55}")
    print(f"  완료! {len(saved)}개 히트맵 생성 → {args.out_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
