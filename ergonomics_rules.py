# ============================================================
# ergonomics_rules.py
# 집중도 판단 규칙 모듈
#
# 3가지 항목을 분석한다:
#   1. 눈 개방도 (EAR)   → 졸음 감지
#   2. 머리 자세          → 고개 돌림·숙임 감지
#   3. 시선 방향          → 화면 이탈 감지
# ============================================================

import cv2
import numpy as np

# ── 일반 모드 임계값 ─────────────────────────────────────────
EAR_THRESHOLD       = 0.25
EAR_CONSEC_FRAMES   = 20     # ≈ 0.67초 @ 30fps

YAW_THRESHOLD       = 30.0
PITCH_THRESHOLD     = 25.0

GAZE_RATIO_MIN      = 0.35
GAZE_RATIO_MAX      = 0.65
GAZE_VERT_MIN       = 0.25   # 홍채가 눈 위쪽 25% 미만 → 위쪽 시선 이탈
GAZE_VERT_MAX       = 0.82   # 홍채가 눈 아래쪽 82% 초과 → 아래쪽 이탈

# ── 시험 감시 모드 임계값 (더 엄격) ──────────────────────────
EXAM_EAR_CONSEC_FRAMES  = 10    # ≈ 0.33초 — 짧은 눈 감음도 포착
EXAM_YAW_THRESHOLD      = 12.0  # 좌우 12도 초과 시 이탈
EXAM_PITCH_THRESHOLD    = 15.0  # 상하 15도 초과 시 이탈
EXAM_GAZE_RATIO_MIN     = 0.38
EXAM_GAZE_RATIO_MAX     = 0.62
EXAM_GAZE_VERT_MIN      = 0.30
EXAM_GAZE_VERT_MAX      = 0.78


# ── 1. 눈 개방도 (Eye Aspect Ratio) ─────────────────────────
# MediaPipe FaceMesh 랜드마크 인덱스
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# 홍채 중심 인덱스 (refine_landmarks=True 시 사용 가능)
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# ── 3D 얼굴 기준점 (머리 자세 추정용) ───────────────────────
# 실제 얼굴 3D 좌표 (단위: mm, 코 끝 원점)
FACE_3D_MODEL = np.array([
    [0.0,     0.0,    0.0  ],   # 코 끝       (1)
    [0.0,  -330.0,  -65.0 ],   # 턱 끝       (152)
    [-225.0, 170.0, -135.0],   # 왼쪽 눈 끝  (263)
    [225.0,  170.0, -135.0],   # 오른쪽 눈 끝(33)
    [-150.0,-150.0, -125.0],   # 왼쪽 입꼬리 (287)
    [150.0, -150.0, -125.0],   # 오른쪽 입꼬리(57)
], dtype=np.float64)

# FaceMesh 에서 대응하는 2D 랜드마크 인덱스
FACE_2D_IDX = [1, 152, 263, 33, 287, 57]


# ── 헬퍼 함수 ─────────────────────────────────────────────────
def _dist(p1, p2):
    """두 점 사이의 유클리드 거리."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _calculate_ear(landmarks, eye_idx):
    """
    EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
    eye_idx : [P1, P2, P3, P4, P5, P6] 순서의 랜드마크 인덱스 6개
    """
    pts = [landmarks[i][:2] for i in eye_idx]   # (x, y) 만 사용
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    h  = _dist(pts[0], pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


# ── 분석 함수 ─────────────────────────────────────────────────

def analyze_ear(face_landmarks, exam_mode: bool = False) -> tuple[str, str, float]:
    """
    눈 개방도(EAR)를 분석하여 졸음 여부를 반환한다.

    Returns:
        status  : 'GOOD' 또는 'EAR_ISSUE'
        message : 화면 표시용 메시지
        ear_avg : 평균 EAR 값
    """
    if face_landmarks is None:
        return 'NO_FACE', '얼굴을 인식할 수 없습니다', 0.0

    left_ear  = _calculate_ear(face_landmarks, LEFT_EYE_IDX)
    right_ear = _calculate_ear(face_landmarks, RIGHT_EYE_IDX)
    ear_avg   = (left_ear + right_ear) / 2.0

    if ear_avg < EAR_THRESHOLD:
        msg = f'눈 감김 감지! EAR: {ear_avg:.2f} (기준 {EAR_THRESHOLD})'
        return 'EAR_ISSUE', msg, ear_avg

    return 'GOOD', f'EAR: {ear_avg:.2f}', ear_avg


def get_ear_consec_threshold(exam_mode: bool = False) -> int:
    return EXAM_EAR_CONSEC_FRAMES if exam_mode else EAR_CONSEC_FRAMES


def analyze_head_pose(face_landmarks, img_w: int, img_h: int, exam_mode: bool = False) -> tuple[str, str, float, float]:
    """
    solvePnP 를 이용해 머리 자세(Yaw/Pitch)를 추정한다.

    Returns:
        status  : 'GOOD' 또는 'HEAD_ISSUE'
        message : 화면 표시용 메시지
        yaw     : 좌우 회전각 (도)
        pitch   : 상하 기울기 (도)
    """
    if face_landmarks is None:
        return 'NO_FACE', '얼굴 없음', 0.0, 0.0

    # 2D 이미지 좌표 수집
    face_2d = np.array(
        [[face_landmarks[i][0], face_landmarks[i][1]] for i in FACE_2D_IDX],
        dtype=np.float64,
    )

    # 카메라 내부 파라미터 (근사값)
    focal  = img_w
    center = (img_w / 2, img_h / 2)
    cam_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 'GOOD', '각도 계산 실패', 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch, yaw, _ = angles

    yaw_thr   = EXAM_YAW_THRESHOLD   if exam_mode else YAW_THRESHOLD
    pitch_thr = EXAM_PITCH_THRESHOLD if exam_mode else PITCH_THRESHOLD

    issues = []
    if abs(yaw) > yaw_thr:
        issues.append(f'좌우 회전 {yaw:.1f}°')
    if abs(pitch) > pitch_thr:
        issues.append(f'상하 기울기 {pitch:.1f}°')

    if issues:
        msg = '머리 방향 이탈: ' + ', '.join(issues)
        return 'HEAD_ISSUE', msg, yaw, pitch

    return 'GOOD', f'Yaw: {yaw:.1f}°  Pitch: {pitch:.1f}°', yaw, pitch


def analyze_gaze(face_landmarks, img_w: int, exam_mode: bool = False) -> tuple[str, str]:
    """
    홍채(iris) 랜드마크를 이용해 시선 방향을 분석한다.
    홍채가 눈 영역 내에서 좌/우로 치우쳐 있으면 화면 이탈로 판단.

    Returns:
        status  : 'GOOD' 또는 'GAZE_ISSUE'
        message : 화면 표시용 메시지
    """
    if face_landmarks is None:
        return 'NO_FACE', '얼굴 없음'

    try:
        # ── 수평 비율 (좌/우 이탈) ───────────────────────────
        l_left   = face_landmarks[LEFT_EYE_IDX[0]][0]
        l_right  = face_landmarks[LEFT_EYE_IDX[3]][0]
        l_iris_x = np.mean([face_landmarks[i][0] for i in LEFT_IRIS_IDX])

        r_left   = face_landmarks[RIGHT_EYE_IDX[0]][0]
        r_right  = face_landmarks[RIGHT_EYE_IDX[3]][0]
        r_iris_x = np.mean([face_landmarks[i][0] for i in RIGHT_IRIS_IDX])

        l_ratio   = (l_iris_x - l_left)  / max(l_right - l_left, 1)
        r_ratio   = (r_iris_x - r_left)  / max(r_right - r_left, 1)
        avg_ratio = (l_ratio + r_ratio) / 2.0

        # ── 수직 비율 (위/아래 이탈) ─────────────────────────
        # LEFT_EYE_IDX: [P1(left), P2(top1), P3(top2), P4(right), P5(bot1), P6(bot2)]
        l_top_y    = min(face_landmarks[LEFT_EYE_IDX[1]][1],
                         face_landmarks[LEFT_EYE_IDX[2]][1])
        l_bot_y    = max(face_landmarks[LEFT_EYE_IDX[4]][1],
                         face_landmarks[LEFT_EYE_IDX[5]][1])
        l_iris_y   = np.mean([face_landmarks[i][1] for i in LEFT_IRIS_IDX])

        r_top_y    = min(face_landmarks[RIGHT_EYE_IDX[1]][1],
                         face_landmarks[RIGHT_EYE_IDX[2]][1])
        r_bot_y    = max(face_landmarks[RIGHT_EYE_IDX[4]][1],
                         face_landmarks[RIGHT_EYE_IDX[5]][1])
        r_iris_y   = np.mean([face_landmarks[i][1] for i in RIGHT_IRIS_IDX])

        l_v_ratio  = (l_iris_y - l_top_y) / max(l_bot_y - l_top_y, 1)
        r_v_ratio  = (r_iris_y - r_top_y) / max(r_bot_y - r_top_y, 1)
        avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

        g_min   = EXAM_GAZE_RATIO_MIN  if exam_mode else GAZE_RATIO_MIN
        g_max   = EXAM_GAZE_RATIO_MAX  if exam_mode else GAZE_RATIO_MAX
        gv_min  = EXAM_GAZE_VERT_MIN   if exam_mode else GAZE_VERT_MIN
        gv_max  = EXAM_GAZE_VERT_MAX   if exam_mode else GAZE_VERT_MAX

        if avg_ratio < g_min or avg_ratio > g_max:
            direction = '왼쪽' if avg_ratio < 0.5 else '오른쪽'
            return 'GAZE_ISSUE', f'시선 이탈 ({direction}) ratio={avg_ratio:.2f}'

        if avg_v_ratio < gv_min:
            return 'GAZE_ISSUE', f'시선 이탈 (위쪽) v_ratio={avg_v_ratio:.2f}'
        if avg_v_ratio > gv_max:
            return 'GAZE_ISSUE', f'시선 이탈 (아래쪽) v_ratio={avg_v_ratio:.2f}'

        return 'GOOD', f'시선 정상 h={avg_ratio:.2f} v={avg_v_ratio:.2f}'

    except (IndexError, ZeroDivisionError):
        return 'GOOD', '시선 분석 불가 (홍채 랜드마크 없음)'


# FaceMesh 귀 주변 랜드마크 인덱스
LEFT_EAR_LM  = 234   # 왼쪽 귀 앞 (tragus 근처)
RIGHT_EAR_LM = 454   # 오른쪽 귀 앞

def analyze_earphone(face_landmarks, frame: np.ndarray) -> tuple:
    """
    YCrCb 적응형 피부색 모델로 이어폰 착용 여부를 감지한다.

    얼굴의 실제 피부색을 샘플링해 YCrCb 색공간에서 피부 범위를 추정하고,
    귀 주변에서 피부 범위를 벗어난 균일한 물체(이어폰)를 탐지한다.
    색상 무관 — 흰 AirPods·검은 유선 이어폰 모두 감지 가능.

    기법 출처: Skin detection in YCbCr color space (Kovac et al., 2003;
               Chai & Ngan, 1999) — OpenCV 기반 구현.
    """
    if face_landmarks is None:
        return False, '', []

    try:
        h, w = frame.shape[:2]

        # ── 피부색 기준 샘플링 (코 끝·양쪽 볼) ──────────────
        # 랜드마크 1=코끝, 50=왼볼, 280=오른볼 (MediaPipe FaceMesh)
        skin_pixels = []
        for lm_idx in (1, 50, 280):
            sx, sy = face_landmarks[lm_idx][0], face_landmarks[lm_idx][1]
            p  = 10
            patch = frame[max(0, sy-p):min(h, sy+p),
                          max(0, sx-p):min(w, sx+p)]
            if patch.size >= 27:
                skin_pixels.append(patch.reshape(-1, 3))

        if not skin_pixels:
            return False, '', []

        skin_arr  = np.concatenate(skin_pixels, axis=0).reshape(-1, 1, 3).astype(np.uint8)
        skin_ycc  = cv2.cvtColor(skin_arr, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(float)
        cr_mean   = float(np.mean(skin_ycc[:, 1]))
        cb_mean   = float(np.mean(skin_ycc[:, 2]))
        cr_std    = max(float(np.std(skin_ycc[:, 1])), 6.0)
        cb_std    = max(float(np.std(skin_ycc[:, 2])), 6.0)

        # ── 귀 영역 비피부 픽셀 분석 ─────────────────────────
        detected_sides = 0
        boxes          = []
        SIGMA          = 3.2   # 피부 허용 범위 (σ). 낮출수록 민감

        for lm_idx in (LEFT_EAR_LM, RIGHT_EAR_LM):
            ex, ey = face_landmarks[lm_idx][0], face_landmarks[lm_idx][1]
            r  = 20
            x1 = max(ex - r, 0); x2 = min(ex + r, w)
            y1 = max(ey - r, 0); y2 = min(ey + r, h)

            roi = frame[y1:y2, x1:x2]
            if roi.size < 27:
                continue

            roi_ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(float)
            cr_roi  = roi_ycc[:, :, 1]
            cb_roi  = roi_ycc[:, :, 2]

            # 피부 범위 벗어난 픽셀 마스크
            non_skin = (
                (np.abs(cr_roi - cr_mean) > cr_std * SIGMA) |
                (np.abs(cb_roi - cb_mean) > cb_std * SIGMA)
            )

            # 가장자리 제외 — 중앙 60% 영역만 사용 (배경·머리카락 차단)
            rh, rw = non_skin.shape
            m_y1 = rh // 5; m_y2 = 4 * rh // 5
            m_x1 = rw // 5; m_x2 = 4 * rw // 5
            non_skin_inner = non_skin[m_y1:m_y2, m_x1:m_x2]
            non_skin_ratio = float(np.mean(non_skin_inner))

            # 비피부 픽셀 내부 밝기 편차: 낮으면 균일한 물체(이어폰)
            ns_vals = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)[non_skin]
            internal_std = float(np.std(ns_vals)) if ns_vals.size >= 5 else 99.0

            if non_skin_ratio > 0.42 and internal_std < 42:
                detected_sides += 1
                boxes.append((x1, y1, x2, y2))

        if detected_sides >= 2:
            return True, '이어폰 착용 의심', boxes
        return False, '', []

    except Exception:
        return False, '', []
