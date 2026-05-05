# ============================================================
# device_detector.py
# YOLOv8 기반 전자기기 감지 모듈
#
# 감지 대상 (COCO 클래스):
#   cell phone, laptop, remote, keyboard, mouse
# ============================================================

import time
import numpy as np

# 감지할 COCO 클래스 이름
DEVICE_CLASSES = {'cell phone', 'laptop', 'remote', 'keyboard', 'mouse'}

# 연속 N 프레임 이상 감지돼야 사용 이벤트로 인정 (깜빡임 방지)
CONSEC_FRAMES_THRESHOLD = 5


class DeviceDetector:
    """
    YOLOv8n 으로 전자기기를 실시간 감지하고 세션 통계를 누적한다.

    사용 예시:
        detector = DeviceDetector()
        boxes, labels = detector.detect(frame)
    """

    def __init__(self, conf: float = 0.45, run_every: int = 2):
        """
        Parameters:
            conf      : 감지 신뢰도 임계값
            run_every : N 프레임마다 1번 추론 (성능 최적화)
        """
        from ultralytics import YOLO
        self.model      = YOLO('yolov8n.pt')   # 자동 다운로드
        self.conf       = conf
        self.run_every  = run_every

        # 프레임 카운터
        self._frame_cnt = 0

        # 연속 감지 카운터
        self._consec    = 0

        # 세션 통계
        self.event_count   = 0      # 사용 이벤트 횟수 (연속 감지 시작 1회)
        self.total_seconds = 0.0    # 총 사용 시간 (초)

        # 현재 감지 상태
        self._in_use       = False
        self._use_start    = 0.0

        # 마지막 추론 결과 캐싱 (격 프레임용)
        self._last_boxes   = []
        self._last_labels  = []

    # ── 추론 ──────────────────────────────────────────────────
    def detect(self, frame: np.ndarray):
        """
        프레임에서 전자기기를 감지한다.

        Returns:
            boxes  : [(x1,y1,x2,y2), ...] 픽셀 좌표
            labels : ['cell phone', ...] 클래스 이름
        """
        self._frame_cnt += 1

        # run_every 프레임마다만 추론 (나머지는 캐시 반환)
        if self._frame_cnt % self.run_every != 0:
            self._update_stats(bool(self._last_boxes))
            return self._last_boxes, self._last_labels

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,
            classes=self._device_class_ids(),
        )

        boxes, labels = [], []
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls)]
                if cls_name in DEVICE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append((x1, y1, x2, y2))
                    labels.append(cls_name)

        self._last_boxes  = boxes
        self._last_labels = labels
        self._update_stats(bool(boxes))
        return boxes, labels

    # ── 통계 업데이트 ─────────────────────────────────────────
    def _update_stats(self, detected: bool):
        if detected:
            self._consec += 1
            if self._consec == CONSEC_FRAMES_THRESHOLD:
                # 새 사용 이벤트 시작
                self._in_use    = True
                self._use_start = time.time()
                self.event_count += 1
        else:
            if self._in_use:
                self.total_seconds += time.time() - self._use_start
                self._in_use = False
            self._consec = 0

    # ── 현재 사용 중 여부 ─────────────────────────────────────
    @property
    def currently_in_use(self) -> bool:
        return self._in_use

    # ── 세션 요약 ─────────────────────────────────────────────
    def summary(self) -> dict:
        total = self.total_seconds
        if self._in_use:
            total += time.time() - self._use_start
        return {
            'device_events' : self.event_count,
            'device_seconds': round(total, 1),
        }

    # ── COCO 클래스 ID 목록 ───────────────────────────────────
    def _device_class_ids(self):
        return [
            i for i, name in self.model.names.items()
            if name in DEVICE_CLASSES
        ]

    def release(self):
        pass
