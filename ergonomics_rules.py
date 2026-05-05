# ============================================================
# ergonomics_rules.py
# 집중도 판단 규칙 모듈
#
# 4가지 항목을 분석한다:
#   1. 눈 개방도 (EAR)   → 졸음 감지
#   2. 머리 자세          → 고개 돌림·숙임 감지
#   3. 시선 방향          → 화면 이탈 감지
#   4. 자세 (어깨)        → 옆으로 기울기 감지
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

SHOULDER_TILT_DEG   = 12.0

# ── 시험 감시 모드 임계값 (더 엄격) ──────────────────────────
EXAM_EAR_CONSEC_FRAMES  = 10    # ≈ 0.33초 — 짧은 눈 감음도 포착
EXAM_YAW_THRESHOLD      = 12.0  # 좌우 12도 초과 시 이탈
EXAM_PITCH_THRESHOLD    = 15.0  # 상하 15도 초과 시 이탈
EXAM_GAZE_RATIO_MIN     = 0.38
EXAM_GAZE_RATIO_MAX     = 0.62


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
        # 왼쪽 눈 가로 범위
        l_left  = face_landmarks[LEFT_EYE_IDX[0]][0]
        l_right = face_landmarks[LEFT_EYE_IDX[3]][0]
        # 왼쪽 홍채 중심 x
        l_iris_x = np.mean([face_landmarks[i][0] for i in LEFT_IRIS_IDX])

        # 오른쪽 눈 가로 범위
        r_left  = face_landmarks[RIGHT_EYE_IDX[0]][0]
        r_right = face_landmarks[RIGHT_EYE_IDX[3]][0]
        r_iris_x = np.mean([face_landmarks[i][0] for i in RIGHT_IRIS_IDX])

        # 홍채가 눈 영역에서 차지하는 비율 (0~1, 0.5 = 정면)
        l_ratio = (l_iris_x - l_left) / max(l_right - l_left, 1)
        r_ratio = (r_iris_x - r_left) / max(r_right - r_left, 1)
        avg_ratio = (l_ratio + r_ratio) / 2.0

        g_min = EXAM_GAZE_RATIO_MIN if exam_mode else GAZE_RATIO_MIN
        g_max = EXAM_GAZE_RATIO_MAX if exam_mode else GAZE_RATIO_MAX

        if avg_ratio < g_min or avg_ratio > g_max:
            direction = '왼쪽' if avg_ratio < 0.5 else '오른쪽'
            return 'GAZE_ISSUE', f'시선 이탈 ({direction}) ratio={avg_ratio:.2f}'

        return 'GOOD', f'시선 정상 ratio={avg_ratio:.2f}'

    except (IndexError, ZeroDivisionError):
        return 'GOOD', '시선 분석 불가 (홍채 랜드마크 없음)'


def analyze_posture(pose_landmarks) -> tuple[str, str]:
    """
    어깨 랜드마크(11, 12번)의 y 좌표 차이로 어깨 기울기를 분석한다.

    Returns:
        status  : 'GOOD' 또는 'POSTURE_ISSUE'
        message : 화면 표시용 메시지
    """
    if pose_landmarks is None:
        return 'GOOD', '자세 감지 불가'   # 어깨 안 보여도 다른 항목으로 판단

    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12

    try:
        l_y = pose_landmarks[LEFT_SHOULDER][1]
        r_y = pose_landmarks[RIGHT_SHOULDER][1]
        l_x = pose_landmarks[LEFT_SHOULDER][0]
        r_x = pose_landmarks[RIGHT_SHOULDER][0]

        # 수평선 대비 어깨 기울기 (도)
        dx   = abs(l_x - r_x)
        dy   = abs(l_y - r_y)
        tilt = np.degrees(np.arctan2(dy, dx)) if dx > 0 else 0.0

        if tilt > SHOULDER_TILT_DEG:
            return 'POSTURE_ISSUE', f'어깨 기울어짐: {tilt:.1f}° (기준 {SHOULDER_TILT_DEG}°)'

        return 'GOOD', f'어깨 기울기: {tilt:.1f}°'

    except (IndexError, TypeError):
        return 'GOOD', '어깨 감지 불가'


# FaceMesh 귀 주변 랜드마크 인덱스
LEFT_EAR_LM  = 234   # 왼쪽 귀 앞 (tragus 근처)
RIGHT_EAR_LM = 454   # 오른쪽 귀 앞

def analyze_earphone(face_landmarks, frame: np.ndarray) -> tuple[bool, str]:
    """
    귀 랜드마크 주변 영역의 밝기 분포로 이어폰 착용 여부를 추정한다.
    정확도가 낮아 보조 지표로만 사용한다.

    Returns:
        detected : 이어폰 의심 여부
        message  : 화면 표시용 메시지
    """
    if face_landmarks is None:
        return False, ''

    try:
        h, w = frame.shape[:2]
        detected_sides = 0

        for lm_idx in (LEFT_EAR_LM, RIGHT_EAR_LM):
            ex, ey = face_landmarks[lm_idx][0], face_landmarks[lm_idx][1]
            r = 18   # 탐색 반경 (픽셀)
            x1 = max(ex - r, 0); x2 = min(ex + r, w)
            y1 = max(ey - r, 0); y2 = min(ey + r, h)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 이어폰(흰색/검정 원형 물체)은 주변 피부 대비 분산이 크다
            std = float(np.std(gray))
            # 피부 영역 평균 밝기
            mean_bright = float(np.mean(gray))

            # 고분산(이물질 존재) + 어두운 소재(전형적 이어폰 색상) 판단
            if std > 38 and mean_bright < 160:
                detected_sides += 1

        if detected_sides >= 1:
            return True, '이어폰 착용 의심'
        return False, ''

    except Exception:
        return False, ''
