# ============================================================
# focus_scorer.py
# 집중도 점수 산출 및 세션 로깅 모듈
#
# 가중치:
#   눈 개방도 (EAR)   35%
#   머리 자세          30%
#   시선 방향          25%
#   자세 (어깨)        10%
# ============================================================

import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np


class FocusScorer:
    """
    프레임마다 4개 지표의 OK/NG 여부를 받아
    가중 평균으로 0~100점의 집중도 점수를 산출한다.
    60프레임(≈2초)마다 CSV 에 기록한다.
    """

    WEIGHTS = {
        'ear'    : 0.35,
        'head'   : 0.30,
        'gaze'   : 0.25,
        'posture': 0.10,
    }

    # 점수 구간 → 상태 라벨
    LEVELS = [
        (80, '집중 중',  (0, 200, 100)),    # 초록
        (60, '보통',    (0, 200, 255)),    # 노랑
        (40, '주의',    (0, 140, 255)),    # 주황
        (0,  '경보',    (0,  60, 230)),    # 빨강
    ]

    LOG_INTERVAL = 60   # N 프레임마다 CSV 저장

    def __init__(self, log_dir: str = 'logs'):
        Path(log_dir).mkdir(exist_ok=True)
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path  = Path(log_dir) / f'session_{ts}.csv'
        self.history   = []      # {'time', 'score', 'ear', 'head', 'gaze', 'posture'}
        self.frame_cnt = 0
        self._init_csv()

    # ── CSV 초기화 ─────────────────────────────────────────────
    def _init_csv(self):
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'score', 'ear_ok', 'head_ok', 'gaze_ok', 'posture_ok'])

    # ── 점수 계산 ──────────────────────────────────────────────
    def update(
        self,
        ear_ok:          bool,
        head_ok:         bool,
        gaze_ok:         bool,
        posture_ok:      bool,
        device_detected: bool = False,
    ) -> int:
        """
        각 지표의 OK/NG 를 입력받아 집중도 점수(0~100)를 반환한다.
        """
        flags = {
            'ear'    : ear_ok,
            'head'   : head_ok,
            'gaze'   : gaze_ok,
            'posture': posture_ok,
        }
        score = round(sum(
            self.WEIGHTS[k] * (100 if v else 0)
            for k, v in flags.items()
        ))

        row = {
            'time'   : time.time(),
            'score'  : score,
            'ear'    : ear_ok,
            'head'   : head_ok,
            'gaze'   : gaze_ok,
            'posture': posture_ok,
            'device' : device_detected,
        }
        self.history.append(row)
        self.frame_cnt += 1

        # 일정 프레임마다 CSV 기록
        if self.frame_cnt % self.LOG_INTERVAL == 0:
            self._write_row(row)

        return score

    def _write_row(self, row: dict):
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%H:%M:%S'),
                row['score'],
                row['ear'],
                row['head'],
                row['gaze'],
                row['posture'],
            ])

    # ── 점수 → 상태 ────────────────────────────────────────────
    def get_level(self, score: int) -> tuple[str, tuple]:
        """점수에 맞는 상태 라벨과 BGR 색상을 반환한다."""
        for threshold, label, color in self.LEVELS:
            if score >= threshold:
                return label, color
        return '경보', (0, 60, 230)

    # ── 최근 N 프레임 평균 ─────────────────────────────────────
    def recent_avg(self, n: int = 30) -> int:
        if not self.history:
            return 100
        recent = self.history[-n:]
        return round(np.mean([r['score'] for r in recent]))

    # ── 세션 요약 ──────────────────────────────────────────────
    def summary(self) -> dict:
        """세션 종료 시 호출해 전체 통계를 반환한다."""
        if not self.history:
            return {}
        scores = [r['score'] for r in self.history]
        return {
            'total_frames'  : len(self.history),
            'avg_score'     : round(np.mean(scores)),
            'min_score'     : int(np.min(scores)),
            'max_score'     : int(np.max(scores)),
            'focus_pct'     : round(sum(1 for s in scores if s >= 70) / len(scores) * 100),
            'drowsy_events' : sum(1 for r in self.history if not r['ear']),
            'head_out'      : sum(1 for r in self.history if not r['head']),
            'gaze_out'      : sum(1 for r in self.history if not r['gaze']),
            'posture_bad'   : sum(1 for r in self.history if not r['posture']),
            'log_path'      : str(self.log_path),
        }
