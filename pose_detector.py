# ============================================================
# pose_detector.py
# MediaPipe FaceMesh(468개 랜드마크) + Pose(33개 랜드마크) 추출 모듈
# ============================================================

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """
    웹캠 프레임에서 얼굴(468개)과 신체(33개) 랜드마크를 추출한다.

    사용 예시:
        detector = PoseDetector()
        img, face_lms, pose_lms = detector.process(frame)
    """

    def __init__(
        self,
        face_confidence: float = 0.6,
        pose_confidence: float = 0.6,
    ):
        self.mp_face  = mp.solutions.face_mesh
        self.mp_pose  = mp.solutions.pose
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # FaceMesh: 468개 3D 얼굴 랜드마크
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # 홍채(iris) 랜드마크 포함
            min_detection_confidence=face_confidence,
            min_tracking_confidence=face_confidence,
        )

        # Pose: 33개 신체 랜드마크 (어깨 감지용)
        self.pose = self.mp_pose.Pose(
            model_complexity=0,             # 속도 우선 (0=Lite)
            min_detection_confidence=pose_confidence,
            min_tracking_confidence=pose_confidence,
        )

    # ----------------------------------------------------------
    def process(self, frame: np.ndarray):
        """
        BGR 프레임을 받아 랜드마크를 추출하고 결과를 반환한다.

        Returns:
            frame       : 랜드마크 연결선이 그려진 원본 프레임
            face_lms    : 얼굴 랜드마크 리스트 [(x,y,z), ...] (없으면 None)
            pose_lms    : 신체 랜드마크 리스트 [(x,y,z), ...] (없으면 None)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        face_result = self.face_mesh.process(rgb)
        pose_result = self.pose.process(rgb)

        rgb.flags.writeable = True
        h, w = frame.shape[:2]

        # ── 얼굴 랜드마크 추출 ─────────────────────────────────
        face_lms = None
        if face_result.multi_face_landmarks:
            lms = face_result.multi_face_landmarks[0].landmark
            face_lms = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in lms]

            # 눈·입 윤곽선만 얇게 그리기 (너무 복잡하면 제거 가능)
            self.mp_draw.draw_landmarks(
                frame,
                face_result.multi_face_landmarks[0],
                self.mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles
                    .get_default_face_mesh_contours_style(),
            )

        # ── 신체 랜드마크 추출 ─────────────────────────────────
        pose_lms = None
        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks.landmark
            pose_lms = [(lm.x, lm.y, lm.z) for lm in lms]

        return frame, face_lms, pose_lms

    def release(self):
        self.face_mesh.close()
        self.pose.close()
