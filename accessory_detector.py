# ============================================================
# accessory_detector.py
# 시험 환경 금지 착용물 감지 모듈
#
# 감지 대상: 모자, 마스크, 선글라스
# 방법: FaceMesh 랜드마크 + 픽셀 색상/밝기 분석
# 안경(투명 렌즈)은 감지하지 않음
# ============================================================

import cv2
import numpy as np
from typing import Optional

# ── FaceMesh 주요 랜드마크 인덱스 ────────────────────────────
FOREHEAD_TOP   = 10    # 이마 위쪽
NOSE_TIP       = 1     # 코 끝
CHIN           = 152   # 턱 끝
LEFT_EYE_OUT   = 33    # 왼쪽 눈 바깥쪽
RIGHT_EYE_OUT  = 263   # 오른쪽 눈 바깥쪽
LEFT_EYE_IN    = 133   # 왼쪽 눈 안쪽
RIGHT_EYE_IN   = 362   # 오른쪽 눈 안쪽

# 연속 감지 확정 프레임 수 (깜빡임 방지)
CONFIRM_FRAMES = 18
RELEASE_STEP   = 2     # 미감지 시 카운터 감소 속도


class AccessoryDetector:
    """
    FaceMesh 랜드마크와 ROI 픽셀 분석으로 모자·마스크·선글라스를 감지한다.
    안경(투명)은 눈 영역 밝기가 정상이므로 오탐하지 않는다.
    """

    WARNINGS = {
        'hat'        : '모자를 벗어주세요',
        'mask'       : '마스크를 벗어주세요',
        'sunglasses' : '선글라스를 벗어주세요',
    }

    def __init__(self):
        self._cnt = {'hat': 0, 'mask': 0, 'sunglasses': 0}
        self.detected = {'hat': False, 'mask': False, 'sunglasses': False}

    # ── 메인 감지 ────────────────────────────────────────────
    def detect(self, frame: np.ndarray, face_lms) -> list[tuple[str, str]]:
        """
        Returns:
            warnings : [('hat', '모자를 벗어주세요'), ...] — 감지된 항목 목록
        """
        if face_lms is None:
            # 얼굴 미감지 시 카운터 유지 (오탐 방지)
            return self._build_result()

        skin_ref = self._get_skin_roi(frame, face_lms)

        results = {
            'hat'       : self._check_hat(frame, face_lms, skin_ref),
            'mask'      : self._check_mask(frame, face_lms, skin_ref),
            'sunglasses': self._check_sunglasses(frame, face_lms, skin_ref),
        }

        for key, raw in results.items():
            if raw:
                self._cnt[key] = min(self._cnt[key] + 1, CONFIRM_FRAMES)
            else:
                self._cnt[key] = max(self._cnt[key] - RELEASE_STEP, 0)
            self.detected[key] = self._cnt[key] >= CONFIRM_FRAMES

        return self._build_result()

    # ── 모자 감지 ─────────────────────────────────────────────
    def _check_hat(self, frame, lms, skin_ref) -> bool:
        """
        이마 위 ROI가 피부보다 어둡고 동시에 균일(저분산)하면 모자로 판단.
        머리카락은 어둡지만 분산이 높으므로 오탐하지 않는다.
        """
        if skin_ref is None:
            return False

        h, w = frame.shape[:2]
        fy = lms[FOREHEAD_TOP][1]
        lx = lms[LEFT_EYE_OUT][0]
        rx = lms[RIGHT_EYE_OUT][0]
        face_w = abs(rx - lx)

        # 이마 바로 위 (머리카락 경계 위쪽) 좁은 영역만 확인
        y1 = max(fy - int(face_w * 0.30), 0)
        y2 = max(fy - int(face_w * 0.08), 0)
        x1 = max(min(lx, rx) + int(face_w * 0.15), 0)
        x2 = min(max(lx, rx) - int(face_w * 0.15), w)

        if y2 <= y1 or x2 <= x1:
            return False

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        roi_gray  = cv2.cvtColor(roi,      cv2.COLOR_BGR2GRAY)
        skin_gray = cv2.cvtColor(skin_ref, cv2.COLOR_BGR2GRAY)

        roi_bright  = float(np.mean(roi_gray))
        skin_bright = float(np.mean(skin_gray))
        roi_std     = float(np.std(roi_gray))   # 낮을수록 균일 → 모자

        roi_hsv  = cv2.cvtColor(roi,      cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.cvtColor(skin_ref, cv2.COLOR_BGR2HSV)
        roi_sat  = float(np.mean(roi_hsv[:, :, 1]))
        skin_sat = float(np.mean(skin_hsv[:, :, 1]))

        # 머리카락: 어둡지만 분산이 큼 (std > 30)
        # 모자    : 어둡고 균일  (std < 22) 또는 뚜렷이 다른 색상
        is_dark    = roi_bright < skin_bright * 0.68
        is_uniform = roi_std < 22
        is_colorful = abs(roi_sat - skin_sat) > 50 and roi_std < 30

        return (is_dark and is_uniform) or is_colorful

    # ── 마스크 감지 ───────────────────────────────────────────
    def _check_mask(self, frame, lms, skin_ref) -> bool:
        """코~턱 영역이 피부보다 균일하고 색상이 다르면 마스크로 판단."""
        if skin_ref is None:
            return False

        h, w = frame.shape[:2]
        nx, ny = lms[NOSE_TIP][0], lms[NOSE_TIP][1]
        cy      = lms[CHIN][1]
        lx      = lms[LEFT_EYE_OUT][0]
        rx      = lms[RIGHT_EYE_OUT][0]
        half_w  = int(abs(rx - lx) * 0.38)

        x1 = max(nx - half_w, 0)
        x2 = min(nx + half_w, w)
        y1 = max(ny + 8, 0)
        y2 = min(cy, h)

        if y2 <= y1 or x2 <= x1:
            return False

        lower = frame[y1:y2, x1:x2]
        if lower.size == 0:
            return False

        lower_std  = float(np.mean(np.std(lower.reshape(-1, 3).astype(float), axis=0)))
        skin_std   = float(np.mean(np.std(skin_ref.reshape(-1, 3).astype(float), axis=0)))
        lower_mean = float(np.mean(lower))
        skin_mean  = float(np.mean(skin_ref))

        # 마스크 판단: 아랫면이 피부보다 균일하고 밝기 차이가 큼
        uniform   = lower_std < 20 and lower_std < skin_std * 0.55
        different = abs(lower_mean - skin_mean) > 28
        return uniform and different

    # ── 선글라스 감지 ─────────────────────────────────────────
    def _check_sunglasses(self, frame, lms, skin_ref) -> bool:
        """
        눈 영역 밝기가 피부 대비 현저히 어두우면 선글라스로 판단.
        일반 안경(투명)은 눈이 그대로 보이므로 오탐하지 않음.
        """
        if skin_ref is None:
            return False

        skin_bright = float(np.mean(cv2.cvtColor(skin_ref, cv2.COLOR_BGR2GRAY)))
        h, w = frame.shape[:2]

        eye_pairs = [
            (LEFT_EYE_OUT,  LEFT_EYE_IN),
            (RIGHT_EYE_IN,  RIGHT_EYE_OUT),
        ]
        brightnesses = []
        for idx_a, idx_b in eye_pairs:
            ex1 = lms[idx_a][0]; ey1 = lms[idx_a][1]
            ex2 = lms[idx_b][0]; ey2 = lms[idx_b][1]
            pad = 10
            rx1 = max(min(ex1, ex2) - pad, 0)
            rx2 = min(max(ex1, ex2) + pad, w)
            ey_mid = (ey1 + ey2) // 2
            ry1 = max(ey_mid - pad, 0)
            ry2 = min(ey_mid + pad, h)
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                brightnesses.append(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))

        if not brightnesses:
            return False

        eye_bright = float(np.mean(brightnesses))
        # 선글라스: 눈 영역이 피부보다 35% 이상 어두움
        return eye_bright < skin_bright * 0.65

    # ── 피부 기준 ROI (이마 중앙) ─────────────────────────────
    def _get_skin_roi(self, frame, lms) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        fx, fy = lms[FOREHEAD_TOP][0], lms[FOREHEAD_TOP][1]
        lx = lms[LEFT_EYE_OUT][0]
        rx = lms[RIGHT_EYE_OUT][0]
        face_w = abs(rx - lx)

        # 이마 중앙 작은 패치 (모자에 가리지 않는 영역)
        cx = (lx + rx) // 2
        x1 = max(cx - int(face_w * 0.18), 0)
        x2 = min(cx + int(face_w * 0.18), w)
        y1 = max(fy + 4, 0)
        y2 = min(fy + 28, h)

        if y2 <= y1 or x2 <= x1:
            return None
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    # ── 결과 정리 ─────────────────────────────────────────────
    def _build_result(self) -> list[tuple[str, str]]:
        return [
            (k, self.WARNINGS[k])
            for k in ('hat', 'mask', 'sunglasses')
            if self.detected[k]
        ]

    @property
    def any_detected(self) -> bool:
        return any(self.detected.values())
