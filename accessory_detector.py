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
HAT_CONFIRM_FRAMES = 36
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

    def __init__(self, hat_detector=None):
        self._cnt = {'hat': 0, 'mask': 0, 'sunglasses': 0}
        self.detected = {'hat': False, 'mask': False, 'sunglasses': False}
        self.hat_detector = hat_detector
        self._last_box = {'hat': None, 'mask': None, 'sunglasses': None}

    # ── 메인 감지 ────────────────────────────────────────────
    def detect(self, frame: np.ndarray, face_lms) -> list[tuple]:
        """
        Returns:
            warnings : [('hat', '모자를 벗어주세요', (x1,y1,x2,y2)), ...] — 감지된 항목 목록
        """
        if face_lms is None:
            return self._build_result()

        skin_ref = self._get_skin_roi(frame, face_lms)

        raw_results = {
            'hat'       : self._check_hat(frame, face_lms, skin_ref),
            'mask'      : self._check_mask(frame, face_lms, skin_ref),
            'sunglasses': self._check_sunglasses(frame, face_lms, skin_ref),
        }

        for key, (raw, box) in raw_results.items():
            confirm_frames = HAT_CONFIRM_FRAMES if key == 'hat' else CONFIRM_FRAMES
            if raw:
                self._cnt[key] = min(self._cnt[key] + 1, confirm_frames)
                if box is not None:
                    self._last_box[key] = box
            else:
                self._cnt[key] = max(self._cnt[key] - RELEASE_STEP, 0)
            self.detected[key] = self._cnt[key] >= confirm_frames

        return self._build_result()

    # ── 모자 감지 ─────────────────────────────────────────────
    def _check_hat(self, frame, lms, skin_ref) -> tuple:
        """
        이마 위 ROI만 어둡다고 모자로 보지 않는다.
        이마 피부가 실제로 가려졌거나, 피부와 확실히 다른 색상의 소재가
        넓게 보일 때만 모자로 판단한다.
        """
        if self.hat_detector is not None:
            yolo_result = self.hat_detector.detect(frame, lms)
            if yolo_result is not None:
                box = self.hat_detector.last_box
                return yolo_result, box

        if skin_ref is None:
            return False, None

        h, w = frame.shape[:2]
        fx, fy = lms[FOREHEAD_TOP][0], lms[FOREHEAD_TOP][1]
        lx = lms[LEFT_EYE_OUT][0]
        rx = lms[RIGHT_EYE_OUT][0]
        face_w = abs(rx - lx)
        if face_w < 20:
            return False, None

        cx = fx

        # 모자 후보 영역: 이마 위 중앙부. 머리카락만으로 확정하지 않도록
        # 아래의 이마 피부 확인과 함께 사용한다.
        y1 = max(fy - int(face_w * 0.36), 0)
        y2 = max(fy - int(face_w * 0.05), 0)
        x1 = max(cx - int(face_w * 0.42), 0)
        x2 = min(cx + int(face_w * 0.42), w)

        hat_box = (x1, y1, x2, y2)

        if y2 <= y1 or x2 <= x1:
            return False, None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, None

        # 이마가 피부처럼 보이면 검은 머리카락만 보고 모자로 확정하지 않는다.
        forehead = self._crop(
            frame,
            cx - int(face_w * 0.22),
            fy + 3,
            cx + int(face_w * 0.22),
            fy + int(face_w * 0.14),
        )
        forehead_visible = (
            forehead is not None and
            self._looks_like_skin(forehead, skin_ref)
        )

        roi_gray  = cv2.cvtColor(roi,      cv2.COLOR_BGR2GRAY)
        skin_gray = cv2.cvtColor(skin_ref, cv2.COLOR_BGR2GRAY)

        roi_bright  = float(np.mean(roi_gray))
        skin_bright = float(np.mean(skin_gray))

        roi_hsv  = cv2.cvtColor(roi,      cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.cvtColor(skin_ref, cv2.COLOR_BGR2HSV)
        skin_sat = float(np.mean(skin_hsv[:, :, 1]))
        skin_hue = float(np.mean(skin_hsv[:, :, 0]))

        hue = roi_hsv[:, :, 0].astype(float)
        sat = roi_hsv[:, :, 1].astype(float)
        hue_diff = np.abs(((hue - skin_hue + 90) % 180) - 90)

        dark_limit = max(20.0, min(skin_bright * 0.58, skin_bright - 30))
        dark_mask = roi_gray < dark_limit
        light_mask = roi_gray > min(245.0, skin_bright + 42)
        color_mask = (
            (sat > max(65, skin_sat + 35)) &
            (hue_diff > 12) &
            (roi_gray < skin_bright * 1.25)
        )
        material_mask = (dark_mask | light_mask | color_mask).astype(np.uint8) * 255
        material_mask = cv2.morphologyEx(
            material_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        mask_bool = material_mask > 0
        coverage = float(np.mean(mask_bool))
        if coverage < 0.34:
            return False, None

        column_coverage = float(np.mean(np.mean(mask_bool, axis=0) > 0.45))
        colorful_candidate = (
            float(np.mean(color_mask)) > 0.28 and
            column_coverage > 0.45
        )
        light_candidate = (
            float(np.mean(light_mask)) > 0.30 and
            column_coverage > 0.45 and
            not forehead_visible
        )
        if colorful_candidate or light_candidate:
            return True, hat_box

        if forehead_visible:
            return False, None

        # 어두운 모자는 이마 쪽을 넓게 가리고 아래 경계가 비교적 수평이다.
        if column_coverage < 0.58:
            return False, None

        bottoms = []
        for x in range(mask_bool.shape[1]):
            ys = np.where(mask_bool[:, x])[0]
            if ys.size:
                bottoms.append(int(ys.max()))
        if len(bottoms) < mask_bool.shape[1] * 0.55:
            return False, None

        boundary_std = float(np.std(bottoms))
        near_bottom = float(np.mean(np.array(bottoms) > mask_bool.shape[0] * 0.48))
        detected = (
            roi_bright < skin_bright * 0.72 and
            boundary_std < max(5.0, mask_bool.shape[0] * 0.16) and
            near_bottom > 0.60
        )
        return detected, (hat_box if detected else None)

    # ── 마스크 감지 ───────────────────────────────────────────
    def _check_mask(self, frame, lms, skin_ref) -> tuple:
        """코~턱 영역이 피부보다 균일하고 색상이 다르면 마스크로 판단."""
        if skin_ref is None:
            return False, None

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
            return False, None

        lower = frame[y1:y2, x1:x2]
        if lower.size == 0:
            return False, None

        lower_std  = float(np.mean(np.std(lower.reshape(-1, 3).astype(float), axis=0)))
        skin_std   = float(np.mean(np.std(skin_ref.reshape(-1, 3).astype(float), axis=0)))
        lower_mean = float(np.mean(lower))
        skin_mean  = float(np.mean(skin_ref))

        # 마스크 판단: 아랫면이 피부보다 균일하고 밝기 차이가 큼
        uniform   = lower_std < 20 and lower_std < skin_std * 0.55
        different = abs(lower_mean - skin_mean) > 28
        detected  = uniform and different
        return detected, ((x1, y1, x2, y2) if detected else None)

    # ── 선글라스 감지 ─────────────────────────────────────────
    def _check_sunglasses(self, frame, lms, skin_ref) -> tuple:
        """
        눈 영역 밝기가 피부 대비 현저히 어두우면 선글라스로 판단.
        일반 안경(투명)은 눈이 그대로 보이므로 오탐하지 않음.
        """
        if skin_ref is None:
            return False, None

        skin_bright = float(np.mean(cv2.cvtColor(skin_ref, cv2.COLOR_BGR2GRAY)))
        h, w = frame.shape[:2]

        eye_pairs = [
            (LEFT_EYE_OUT,  LEFT_EYE_IN),
            (RIGHT_EYE_IN,  RIGHT_EYE_OUT),
        ]
        brightnesses = []
        all_xs, all_ys = [], []
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
                all_xs += [rx1, rx2]
                all_ys += [ry1, ry2]

        if not brightnesses:
            return False, None

        eye_bright = float(np.mean(brightnesses))
        detected = eye_bright < skin_bright * 0.65
        if detected and all_xs:
            box = (min(all_xs), min(all_ys), max(all_xs), max(all_ys))
        else:
            box = None
        return detected, box

    # ── 피부 기준 ROI (이마 중앙) ─────────────────────────────
    def _get_skin_roi(self, frame, lms) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        fx, fy = lms[FOREHEAD_TOP][0], lms[FOREHEAD_TOP][1]
        lx = lms[LEFT_EYE_OUT][0]
        rx = lms[RIGHT_EYE_OUT][0]
        face_w = abs(rx - lx)

        patches = []

        # 이마 중앙 작은 패치
        cx = (lx + rx) // 2
        forehead = self._crop(
            frame,
            cx - int(face_w * 0.18),
            fy + 4,
            cx + int(face_w * 0.18),
            fy + 28,
        )
        if forehead is not None:
            patches.append(forehead.reshape(-1, 3))

        # 눈 아래 양쪽 볼 패치. 모자가 이마를 가려도 피부 기준을 잡기 위함.
        nx, ny = lms[NOSE_TIP][0], lms[NOSE_TIP][1]
        cheek_y1 = int(ny - face_w * 0.14)
        cheek_y2 = int(ny + face_w * 0.02)
        for cheek_cx in (
            int(nx - face_w * 0.22),
            int(nx + face_w * 0.22),
        ):
            cheek = self._crop(
                frame,
                cheek_cx - int(face_w * 0.10),
                cheek_y1,
                cheek_cx + int(face_w * 0.10),
                cheek_y2,
            )
            if cheek is not None:
                patches.append(cheek.reshape(-1, 3))

        if not patches:
            return None
        return np.concatenate(patches, axis=0).reshape(-1, 1, 3)

    @staticmethod
    def _crop(frame, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        if y2 <= y1 or x2 <= x1:
            return None
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    @staticmethod
    def _looks_like_skin(roi: np.ndarray, skin_ref: np.ndarray) -> bool:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        skin_gray = cv2.cvtColor(skin_ref, cv2.COLOR_BGR2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.cvtColor(skin_ref, cv2.COLOR_BGR2HSV)

        bright_diff = abs(float(np.mean(roi_gray)) - float(np.mean(skin_gray)))
        sat_diff = abs(float(np.mean(roi_hsv[:, :, 1])) - float(np.mean(skin_hsv[:, :, 1])))
        return bright_diff < 28 and sat_diff < 35

    # ── 결과 정리 ─────────────────────────────────────────────
    def _build_result(self) -> list[tuple]:
        return [
            (k, self.WARNINGS[k], self._last_box.get(k))
            for k in ('hat', 'mask', 'sunglasses')
            if self.detected[k]
        ]

    @property
    def any_detected(self) -> bool:
        return any(self.detected.values())
