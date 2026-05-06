# ============================================================
# hat_detector.py
# YOLO 기반 모자 감지 보조 모듈
#
# models/hat_detector.pt 파일이 있을 때만 활성화된다.
# 시험 모드에서만 생성해 사용한다.
# ============================================================

from pathlib import Path
from typing import Optional

import numpy as np


HAT_LABELS = {
    'cap',
    'hat',
    'hardhat',
    'hard_hat',
    'helmet',
    'safety_helmet',
}
NO_HAT_LABELS = {
    'no_cap',
    'no-cap',
    'nocap',
    'no hat',
    'no_hat',
    'no-hat',
    'no hardhat',
    'no-hardhat',
    'no_helmet',
    'without_cap',
    'without_hat',
}


class YoloHatDetector:
    """
    cap/no_hat 또는 hat/no_hat 계열 YOLO 모델을 로드해 모자 착용 여부를 판단한다.

    detect() 반환값:
        True  : 모자 감지
        False : no_hat 감지 또는 명확한 미착용
        None  : 모델 없음/비활성/판단 불가
    """

    def __init__(
        self,
        model_path: str = 'models/hat_detector.pt',
        conf: float = 0.42,
        run_every: int = 3,
    ):
        self.model_path = Path(model_path)
        self.conf = conf
        self.run_every = max(1, int(run_every))
        self.enabled = False
        self.model = None
        self.names: dict[int, str] = {}
        self._frame_cnt = 0
        self._last_result: Optional[bool] = None
        self._last_box: Optional[tuple] = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            print(f'[YoloHatDetector] 모델 없음 — ROI 모자 감지 사용: {self.model_path}')
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.names = {
                int(k): self._normalize_label(v)
                for k, v in self.model.names.items()
            }
            self.enabled = True
            print(f'[YoloHatDetector] YOLO 모자 모델 로드: {self.model_path}')
            print(f'[YoloHatDetector] 클래스: {self.names}')
        except Exception as e:
            print(f'[YoloHatDetector] 모델 로드 실패 — ROI 모자 감지 사용: {e}')
            self.model = None
            self.enabled = False

    def detect(self, frame: np.ndarray, face_lms=None) -> Optional[bool]:
        if not self.enabled or self.model is None:
            return None

        self._frame_cnt += 1
        if self._frame_cnt % self.run_every != 0:
            return self._last_result

        try:
            results = self.model(frame, conf=self.conf, verbose=False)
        except Exception as e:
            print(f'[YoloHatDetector] 추론 실패: {e}')
            self._last_result = None
            return None

        best_hat = 0.0
        best_no_hat = 0.0
        best_box = None

        for r in results:
            for box in r.boxes:
                label = self.names.get(int(box.cls), '')
                conf = float(box.conf)
                if not self._is_relevant_box(box, face_lms, frame.shape):
                    continue
                if label in HAT_LABELS:
                    if conf > best_hat:
                        best_hat = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        best_box = (x1, y1, x2, y2)
                elif label in NO_HAT_LABELS:
                    best_no_hat = max(best_no_hat, conf)

        if best_hat == 0.0 and best_no_hat == 0.0:
            self._last_result = False
            self._last_box = None
        else:
            self._last_result = best_hat > best_no_hat
            self._last_box = best_box if self._last_result else None
        return self._last_result

    @property
    def last_box(self) -> Optional[tuple]:
        return self._last_box

    @staticmethod
    def _normalize_label(label) -> str:
        return str(label).strip().lower().replace(' ', '_')

    @staticmethod
    def _is_relevant_box(box, face_lms, frame_shape) -> bool:
        if face_lms is None:
            return True

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)

        xs = [lm[0] for lm in face_lms]
        ys = [lm[1] for lm in face_lms]
        fx1, fx2 = min(xs), max(xs)
        fy1, fy2 = min(ys), max(ys)
        face_w = max(fx2 - fx1, 1)
        face_h = max(fy2 - fy1, 1)

        if box_w > face_w * 1.90 or box_h > face_h * 0.90:
            return False
        if box_w < face_w * 0.25 or box_h < face_h * 0.08:
            return False

        hx1 = max(fx1 - face_w * 0.35, 0)
        hx2 = min(fx2 + face_w * 0.35, w)
        hy1 = max(fy1 - face_h * 0.70, 0)
        hy2 = min(fy1 + face_h * 0.18, h)

        ix1 = max(x1, hx1)
        iy1 = max(y1, hy1)
        ix2 = min(x2, hx2)
        iy2 = min(y2, hy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False

        overlap = (ix2 - ix1) * (iy2 - iy1)
        box_area = box_w * box_h
        head_area = max((hx2 - hx1) * (hy2 - hy1), 1.0)

        return (
            hx1 <= bx <= hx2 and
            hy1 <= by <= hy2 and
            overlap / box_area > 0.20 and
            overlap / head_area > 0.08
        )
