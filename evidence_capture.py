# ============================================================
# evidence_capture.py
# 부정행위 의심 시 증거 사진 저장 모듈
# ============================================================

import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# 동일 유형 쿨다운 (초) — 짧은 시간에 같은 유형 중복 저장 방지
COOLDOWN = {
    'device'    : 4.0,
    'hat'       : 6.0,
    'mask'      : 6.0,
    'sunglasses': 6.0,
    'earphone'  : 8.0,
}

LABEL_KO = {
    'device'    : '전자기기 감지',
    'hat'       : '모자 착용',
    'mask'      : '마스크 착용',
    'sunglasses': '선글라스 착용',
    'earphone'  : '이어폰 착용 의심',
}

MAX_SHOTS = 60   # 세션당 최대 저장 장수


class EvidenceCapture:
    """
    부정행위 의심 프레임을 logs/evidence_<세션ID>/ 폴더에 저장한다.
    """

    def __init__(self, session_id: str, log_dir: str = 'logs'):
        self.evidence_dir = Path(log_dir) / f'evidence_{session_id}'
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._last_time: dict[str, float] = {k: 0.0 for k in COOLDOWN}
        self._count = 0
        self.saved: list[dict] = []   # {'path', 'type', 'label', 'time'}

    # ── 저장 시도 ─────────────────────────────────────────────
    def try_save(self, frame: np.ndarray, ev_type: str) -> bool:
        """
        조건을 만족하면 프레임을 저장하고 True 를 반환한다.
        """
        if self._count >= MAX_SHOTS:
            return False
        now = time.time()
        cd  = COOLDOWN.get(ev_type, 5.0)
        if now - self._last_time.get(ev_type, 0.0) < cd:
            return False

        self._last_time[ev_type] = now
        label = LABEL_KO.get(ev_type, ev_type)

        # 저장용 프레임: 라벨 텍스트 삽입
        img = frame.copy()
        ts_str = datetime.now().strftime('%H:%M:%S')
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 140), -1)
        cv2.putText(img, f'[{ts_str}] {label}',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)

        fname = self.evidence_dir / f'{self._count:03d}_{ts_str.replace(":", "")}_{ev_type}.jpg'
        cv2.imwrite(str(fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        self.saved.append({
            'path' : str(fname),
            'type' : ev_type,
            'label': label,
            'time' : ts_str,
        })
        self._count += 1
        return True

    @property
    def count(self) -> int:
        return self._count

    @property
    def directory(self) -> str:
        return str(self.evidence_dir)
