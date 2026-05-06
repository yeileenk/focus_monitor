# ============================================================
# exam_registration.py
# 시험 모드 시작 시 5자세 등록 촬영 모듈
#
# 등록 순서:
#   1. 정면
#   2. 턱 들기
#   3. 고개 숙이기
#   4. 왼쪽 귀 보여주기
#   5. 오른쪽 귀 보여주기
#
# 모자·선글라스·마스크·이어폰·전자기기 감지 시 촬영 불가.
# 등록 완료 후 세션 중 얼굴 히스토그램으로 인물 교체를 감시한다.
# ============================================================

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from pathlib import Path
from typing import Optional

rcParams['font.family'] = ['AppleGothic', 'Apple SD Gothic Neo', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


POSES = [
    {
        'key'        : 'front',
        'label'      : '정면',
        'step'       : '1 / 5',
        'instruction': '정면을 바라보세요',
        'sub'        : '눈높이를 맞추고 카메라를 자연스럽게 응시',
        'check'      : lambda yaw, pitch: abs(yaw) < 12 and abs(pitch) < 10,
    },
    {
        'key'        : 'chin_up',
        'label'      : '턱 들기',
        'step'       : '2 / 5',
        'instruction': '턱을 들어 위를 바라보세요',
        'sub'        : '이마가 카메라에 보일 정도로',
        'check'      : lambda yaw, pitch: pitch < -8,
    },
    {
        'key'        : 'chin_down',
        'label'      : '고개 숙이기',
        'step'       : '3 / 5',
        'instruction': '고개를 숙여 아래를 바라보세요',
        'sub'        : '정수리가 살짝 보일 정도로',
        'check'      : lambda yaw, pitch: pitch > 8,
    },
    {
        'key'        : 'left_ear',
        'label'      : '왼쪽 귀',
        'step'       : '4 / 5',
        'instruction': '왼쪽 귀가 보이게 고개를 돌리세요',
        'sub'        : '오른쪽으로 45도 — 귀가 완전히 보여야 합니다',
        'check'      : lambda yaw, pitch: yaw < -20,
    },
    {
        'key'        : 'right_ear',
        'label'      : '오른쪽 귀',
        'step'       : '5 / 5',
        'instruction': '오른쪽 귀가 보이게 고개를 돌리세요',
        'sub'        : '왼쪽으로 45도 — 귀가 완전히 보여야 합니다',
        'check'      : lambda yaw, pitch: yaw > 20,
    },
]

HOLD_FRAMES        = 20     # 자세 유지 확인 프레임 수
FACE_SIM_INTERVAL  = 90     # 세션 중 얼굴 비교 주기 (프레임)
FACE_SIM_THRESHOLD = 0.25   # 히스토그램 유사도 최소값 (미만이면 인물 불일치 의심)

_ACC_KO = {'hat': '모자', 'sunglasses': '선글라스', 'mask': '마스크'}


class ExamRegistration:
    """
    5자세 등록 촬영을 진행하고, 세션 중 얼굴 유사도를 비교한다.
    """

    def __init__(self, session_id: str, log_dir: str = 'logs'):
        self.session_id  = session_id
        self.save_dir    = Path(log_dir) / f'registration_{session_id}'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ref_hists : list[np.ndarray] = []
        self.ref_photos: list[str]        = []
        self._completed  = False

    @property
    def completed(self) -> bool:
        return self._completed

    # ── 등록 촬영 메인 ───────────────────────────────────────
    def run(
        self,
        cap,
        detector,
        accessory_detector,
        device_detector,
        feedback,
        analyze_head_pose_fn,
        analyze_earphone_fn,
        win_title: str = 'exam_monitor',
    ) -> bool:
        """
        5자세 등록을 진행한다.
        완료 → True, 취소(q 키 또는 종료 버튼) → False.
        """
        quit_flag  = [False]
        frame_size = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if feedback.is_quit_button_clicked(x, y, frame_size[0], frame_size[1]):
                    quit_flag[0] = True

        cv2.setMouseCallback(win_title, _on_mouse)
        self._show_intro(cap, feedback, win_title)

        pose_idx     = 0
        hold_counter = 0

        while pose_idx < len(POSES):
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            frame_size[0], frame_size[1] = w, h

            frame, face_lms = detector.process(frame)
            pose = POSES[pose_idx]

            violations: list[str] = []
            yaw = pitch = 0.0
            pose_ok = False

            if face_lms is None:
                violations.append('얼굴 미인식')
            else:
                _, _, yaw, pitch = analyze_head_pose_fn(face_lms, w, h, True)
                pose_ok = pose['check'](yaw, pitch)

                if accessory_detector:
                    for item in accessory_detector.detect(frame, face_lms):
                        violations.append(_ACC_KO.get(item[0], item[0]))

                ep, _, _ = analyze_earphone_fn(face_lms, frame)
                if ep:
                    violations.append('이어폰')

                dev_boxes, _, _ = device_detector.detect(frame)
                if dev_boxes:
                    violations.append('전자기기')

            if violations:
                hold_counter = 0
            elif pose_ok:
                hold_counter = min(hold_counter + 1, HOLD_FRAMES)
            else:
                hold_counter = max(hold_counter - 1, 0)

            frame = self._draw_ui(frame, feedback, pose, violations, hold_counter, yaw, pitch)
            frame = feedback.draw_quit_button(frame)
            cv2.imshow(win_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or quit_flag[0]:
                return False

            if hold_counter >= HOLD_FRAMES and not violations:
                self._capture(frame, face_lms, pose_idx, pose['key'])
                self._flash(frame, feedback, win_title)
                pose_idx    += 1
                hold_counter = 0

        self._completed = True
        return True

    # ── 세션 중 얼굴 비교 ────────────────────────────────────
    def check_identity(
        self,
        frame: np.ndarray,
        face_lms,
        frame_cnt: int,
    ) -> bool:
        """
        FACE_SIM_INTERVAL 프레임마다 등록 히스토그램과 비교.
        유사도가 낮으면 True (인물 불일치 의심).
        """
        if not self.ref_hists or face_lms is None:
            return False
        if frame_cnt % FACE_SIM_INTERVAL != 0:
            return False
        hist = self._face_hist(frame, face_lms)
        if hist is None:
            return False
        sim = max(
            cv2.compareHist(hist, ref, cv2.HISTCMP_CORREL)
            for ref in self.ref_hists
        )
        return sim < FACE_SIM_THRESHOLD

    # ── 내부 헬퍼 ────────────────────────────────────────────

    def _capture(self, frame: np.ndarray, face_lms, idx: int, key: str):
        path = str(self.save_dir / f'reg_{idx+1:02d}_{key}.jpg')
        cv2.imwrite(path, frame)
        self.ref_photos.append(path)
        hist = self._face_hist(frame, face_lms)
        if hist is not None:
            self.ref_hists.append(hist)
        print(f'[등록] {idx+1}/5 촬영 완료: {path}')

    def _flash(self, frame: np.ndarray, feedback, win_title: str):
        h, w    = frame.shape[:2]
        white   = np.ones_like(frame) * 240
        blended = cv2.addWeighted(frame, 0.3, white, 0.7, 0)
        blended = feedback.put_korean_text(
            blended, '촬영 완료!',
            (w // 2 - 65, h // 2 - 24), font_size=34,
            color=(0, 140, 55), bg_color=(220, 240, 220),
        )
        cv2.imshow(win_title, blended)
        cv2.waitKey(600)

    def _show_intro(self, cap, feedback, win_title: str):
        """등록 시작 전 3초 카운트다운."""
        for count in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (10, 20, 45), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            frame = feedback.put_korean_text(
                frame, '[시험 모드]  등록 촬영을 시작합니다',
                (w // 2 - 200, h // 2 - 80), font_size=22, color=(255, 215, 40),
            )
            frame = feedback.put_korean_text(
                frame, '모자 · 선글라스 · 이어폰 · 전자기기 없이 촬영하세요',
                (w // 2 - 255, h // 2 - 38), font_size=17, color=(200, 215, 235),
            )
            frame = feedback.put_korean_text(
                frame, f'{count}',
                (w // 2 - 18, h // 2 + 10), font_size=52, color=(255, 255, 100),
            )
            frame = feedback.put_korean_text(
                frame, 'q 키로 취소',
                (w // 2 - 55, h // 2 + 80), font_size=14, color=(130, 150, 180),
            )
            cv2.imshow(win_title, frame)
            cv2.waitKey(1000)

    # ── 등록 사진 리포트 ─────────────────────────────────────
    def generate_registration_report(self, output_dir: str = 'logs') -> Optional[str]:
        """
        등록된 5장 사진을 포즈 레이블과 함께 한 창에 표시하는 PNG를 생성한다.
        세션 종료 후 evidence 뷰어와 함께 자동으로 열린다.
        """
        if not self.ref_photos:
            return None

        n = len(self.ref_photos)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5))
        fig.patch.set_facecolor('#0F1B35')
        fig.suptitle('시험 등록 촬영 — 5자세 확인',
                     fontsize=17, color='white', fontweight='bold', y=1.01)

        if n == 1:
            axes = [axes]

        for i, (ax, photo_path) in enumerate(zip(axes, self.ref_photos)):
            ax.set_facecolor('#1A2D50')
            try:
                img_bgr = cv2.imread(photo_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            except Exception:
                ax.text(0.5, 0.5, '사진 없음', transform=ax.transAxes,
                        ha='center', va='center', color='white', fontsize=12)

            pose  = POSES[i] if i < len(POSES) else {}
            step  = pose.get('step', f'{i+1}/5')
            label = pose.get('label', f'포즈 {i+1}')
            instr = pose.get('instruction', '')

            ax.set_title(f'{step}  {label}\n{instr}',
                         color='#FBBF24', fontsize=11, pad=6,
                         fontweight='bold')
            ax.axis('off')

            # 테두리
            for spine in ax.spines.values():
                spine.set_edgecolor('#334155')
                spine.set_linewidth(1.5)

        plt.tight_layout(pad=1.2)
        out_path = Path(output_dir) / f'reg_report_{self.session_id}.png'
        fig.savefig(str(out_path), dpi=110, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[등록] 등록 사진 리포트 저장: {out_path}')
        return str(out_path)

    @staticmethod
    def _face_hist(frame: np.ndarray, face_lms) -> Optional[np.ndarray]:
        try:
            xs = [lm[0] for lm in face_lms]
            ys = [lm[1] for lm in face_lms]
            h, w = frame.shape[:2]
            x1 = max(min(xs) - 10, 0)
            x2 = min(max(xs) + 10, w)
            y1 = max(min(ys) - 10, 0)
            y2 = min(max(ys) + 10, h)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist
        except Exception:
            return None

    @staticmethod
    def _draw_ui(frame, feedback, pose, violations, hold_counter, yaw, pitch):
        h, w = frame.shape[:2]

        # 상단 헤더
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 108), (10, 20, 45), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        frame = feedback.put_korean_text(
            frame, f'[시험 등록]   {pose["step"]}  —  {pose["label"]}',
            (18, 10), font_size=21, color=(255, 215, 40),
        )
        frame = feedback.put_korean_text(
            frame, pose['instruction'],
            (18, 46), font_size=20, color=(255, 255, 255),
        )
        frame = feedback.put_korean_text(
            frame, pose['sub'],
            (18, 80), font_size=13, color=(140, 175, 215),
        )

        if violations:
            msg = '제거 후 다시 시도:   ' + ',   '.join(dict.fromkeys(violations))
            frame = feedback.put_korean_text(
                frame, msg,
                (18, h - 74), font_size=15, color=(60, 60, 255), bg_color=(0, 0, 0),
            )
            frame = feedback.put_korean_text(
                frame, '위반 물품이 감지되어 촬영할 수 없습니다',
                (18, h - 44), font_size=14, color=(90, 90, 230),
            )
        else:
            status = '자세 유지 중...' if hold_counter > 0 else '자세를 잡아주세요'
            clr    = (0, 210, 120) if hold_counter > 0 else (150, 175, 200)
            frame  = feedback.put_korean_text(
                frame, status, (18, h - 52), font_size=15, color=clr,
            )
            bx1, by1 = 18, h - 28
            bx2, by2 = w - 18, h - 8
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 50, 70), -1)
            filled = int((hold_counter / HOLD_FRAMES) * (bx2 - bx1))
            if filled > 0:
                cv2.rectangle(frame, (bx1, by1), (bx1 + filled, by2), (0, 210, 100), -1)

        # 각도 디버그 (우하단)
        frame = feedback.put_korean_text(
            frame, f'yaw {yaw:+.0f}°  pitch {pitch:+.0f}°',
            (w - 200, h - 12), font_size=11, color=(80, 100, 130),
        )
        return frame
