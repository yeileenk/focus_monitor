# ============================================================
# feedback_handler.py
# 사용자 피드백 모듈
#
# 기능:
#   - 화면에 집중도 점수 오버레이 표시 (한글 포함)
#   - 집중도 경보 시 화면 테두리 빨간색
#   - gTTS + pygame 으로 한국어 음성 알림
#   - 동일 경보 중복 재생 방지 (쿨다운 10초)
# ============================================================

import os
import time
import threading
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


class FeedbackHandler:
    """
    화면 오버레이(한글) 와 음성 알림을 제공한다.
    pygame.mixer 는 처음 인스턴스 생성 시 한 번만 초기화한다.
    """

    COOLDOWN_SEC  = 10.0   # 음성 알림 재발생 최소 간격 (초)
    ALERT_SCORE   = 40     # 이 점수 미만이면 경보
    BORDER_COLOR  = (0, 50, 220)   # 경보 테두리 색 (BGR 빨강)
    BORDER_WIDTH  = 12

    # 음성 메시지 매핑
    VOICE_MSGS = {
        'EAR_ISSUE'     : '눈이 감기고 있습니다. 졸음을 주의하세요.',
        'HEAD_ISSUE'    : '고개가 돌아갔습니다. 화면을 바라봐 주세요.',
        'GAZE_ISSUE'    : '시선이 화면을 벗어났습니다.',
        'LOW_SCORE'     : '집중도가 낮아졌습니다. 잠시 스트레칭해보세요.',
    }

    def __init__(self):
        self._last_voice_time  : float = 0.0
        self._last_voice_key   : str   = ''
        self._mixer_ready      : bool  = False
        self._init_mixer()

        # 한글 폰트 경로 탐색 (없으면 기본 폰트 사용)
        self.font_path = self._find_korean_font()

    # ── pygame mixer 초기화 ───────────────────────────────────
    def _init_mixer(self):
        try:
            import pygame
            pygame.mixer.init()
            self._mixer_ready = True
        except Exception as e:
            print(f'[FeedbackHandler] pygame 초기화 실패 (음성 비활성화): {e}')

    # ── 한글 폰트 탐색 ────────────────────────────────────────
    @staticmethod
    def _find_korean_font() -> Optional[str]:
        candidates = [
            # Windows
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/NanumGothic.ttf',
            # macOS
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            # Linux
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    # ── 한글 텍스트 → OpenCV 프레임 ──────────────────────────
    def put_korean_text(
        self,
        frame: np.ndarray,
        text:  str,
        pos:   tuple[int, int],
        font_size: int = 22,
        color: tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """PIL 로 한글을 렌더링한 뒤 OpenCV 프레임에 붙인다."""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)

        try:
            font = (
                ImageFont.truetype(self.font_path, font_size)
                if self.font_path
                else ImageFont.load_default()
            )
        except Exception:
            font = ImageFont.load_default()

        x, y = pos
        # 배경 박스
        if bg_color is not None:
            bbox = draw.textbbox((x, y), text, font=font)
            pad  = 4
            draw.rectangle(
                [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                fill=bg_color,
            )
        draw.text((x, y), text, font=font, fill=(color[0], color[1], color[2]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── 스터디 타이머 표시 ───────────────────────────────────
    def draw_study_timer(
        self,
        frame: np.ndarray,
        time_left: float,
        total: float,
    ) -> np.ndarray:
        """우상단에 남은 시간 + 진행 바를 표시한다."""
        h, w = frame.shape[:2]

        rem   = int(time_left)
        hh    = rem // 3600
        mm    = (rem % 3600) // 60
        ss    = rem % 60
        ts    = f'{hh:02d}:{mm:02d}:{ss:02d}' if hh > 0 else f'{mm:02d}:{ss:02d}'

        # 진행률
        ratio = 1.0 - (time_left / total) if total > 0 else 1.0

        # 배경 박스
        bx1, by1, bx2, by2 = w - 180, 36, w - 8, 82
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), 1)

        # 남은 시간 텍스트
        color = (80, 220, 80) if time_left > total * 0.2 else (60, 100, 255)
        frame = self.put_korean_text(frame, ts, (bx1 + 14, by1 + 6),
                                     font_size=22, color=color)

        # 진행 바
        bar_x1 = bx1 + 6
        bar_y1 = by2 - 10
        bar_w  = bx2 - bx1 - 12
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, by2 - 4),
                      (50, 50, 50), -1)
        filled = int(bar_w * ratio)
        if filled > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + filled, by2 - 4), color, -1)
        return frame

    # ── 종료 버튼 ────────────────────────────────────────────
    QUIT_BTN_W = 90
    QUIT_BTN_H = 34
    QUIT_BTN_MARGIN = 12

    def draw_quit_button(self, frame: np.ndarray) -> np.ndarray:
        """우하단에 종료 버튼을 그린다."""
        h, w = frame.shape[:2]
        m = self.QUIT_BTN_MARGIN
        x1 = w - self.QUIT_BTN_W - m
        y1 = h - self.QUIT_BTN_H - m
        x2 = w - m
        y2 = h - m

        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
        frame = self.put_korean_text(
            frame, '종료',
            (x1 + 22, y1 + 6),
            font_size=16, color=(220, 220, 220),
        )
        return frame

    def is_quit_button_clicked(self, x: int, y: int, frame_w: int, frame_h: int) -> bool:
        """클릭 좌표가 종료 버튼 영역 안에 있는지 확인한다."""
        m = self.QUIT_BTN_MARGIN
        x1 = frame_w - self.QUIT_BTN_W - m
        y1 = frame_h - self.QUIT_BTN_H - m
        x2 = frame_w - m
        y2 = frame_h - m
        return x1 <= x <= x2 and y1 <= y <= y2

    # ── 화면 오버레이 ─────────────────────────────────────────
    def draw_overlay(
        self,
        frame:    np.ndarray,
        score:    int,
        level:    str,
        color:    tuple[int, int, int],
        ear_msg:  str,
        head_msg: str,
        gaze_msg: str,
        ear_ok:   bool,
        head_ok:  bool,
        gaze_ok:  bool,
        exam_mode: bool = False,
    ) -> np.ndarray:
        """
        프레임에 집중도 정보를 그린다.
        """
        h, w = frame.shape[:2]

        # ── 시험 모드 상단 라벨 ───────────────────────────────
        if exam_mode:
            cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 160), -1)
            frame = self.put_korean_text(
                frame, '[시험 감시 모드 활성]', (w // 2 - 110, 4),
                font_size=15, color=(255, 255, 255),
            )

        # ── 경보 테두리 ───────────────────────────────────────
        if score < self.ALERT_SCORE:
            cv2.rectangle(
                frame,
                (0, 0), (w-1, h-1),
                self.BORDER_COLOR, self.BORDER_WIDTH,
            )

        # ── 상단 반투명 배경 박스 ─────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # ── 점수 표시 ─────────────────────────────────────────
        frame = self.put_korean_text(
            frame, f'집중도: {score}점', (16, 10),
            font_size=32, color=color,
        )
        frame = self.put_korean_text(
            frame, f'● {level}', (16, 52),
            font_size=20, color=color,
        )

        # ── 3개 지표 상태 (우측 상단) ─────────────────────────
        indicators = [
            ('눈', ear_ok),
            ('머리', head_ok),
            ('시선', gaze_ok),
        ]
        for i, (label, ok) in enumerate(indicators):
            dot_color = (0, 210, 80) if ok else (0, 60, 220)
            frame = self.put_korean_text(
                frame,
                f'{"✓" if ok else "✗"} {label}',
                (w - 170 + i * 55, 14),
                font_size=16, color=dot_color,
            )

        # ── 하단 경고 메시지 ──────────────────────────────────
        alerts = []
        if not ear_ok:  alerts.append(ear_msg)
        if not head_ok: alerts.append(head_msg)
        if not gaze_ok: alerts.append(gaze_msg)

        for j, msg in enumerate(alerts[:2]):
            frame = self.put_korean_text(
                frame, f'⚠ {msg}',
                (10, h - 90 + j * 36),
                font_size=17,
                color=(20, 80, 255),
                bg_color=(0, 0, 0),
            )

        return frame

    # ── 전자기기 바운딩 박스 ──────────────────────────────────
    def draw_device_boxes(
        self,
        frame: np.ndarray,
        boxes: list,
        labels: list,
        confs: list = None,
    ) -> np.ndarray:
        """감지된 전자기기에 빨간 박스와 라벨·신뢰도를 그린다."""
        for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes, labels)):
            conf = confs[i] if confs else None
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
            conf_str = f' {conf:.0%}' if conf is not None else ''
            frame = self.put_korean_text(
                frame, f'[기기] {label}{conf_str}',
                (x1, max(y1 - 26, 0)),
                font_size=15, color=(60, 60, 255), bg_color=(0, 0, 0),
            )
        return frame

    # ── 착용물·이어폰 바운딩 박스 ─────────────────────────────
    def draw_accessory_boxes(
        self,
        frame: np.ndarray,
        acc_warnings: list,
        earphone_boxes: list = None,
    ) -> np.ndarray:
        """모자·마스크·선글라스·이어폰의 바운딩 박스와 레이블을 그린다."""
        BOX_COLORS = {
            'hat'       : (0, 140, 255),   # 주황
            'mask'      : (0, 220, 220),   # 노랑
            'sunglasses': (200,  50, 200), # 보라
        }
        for item in acc_warnings:
            acc_type, msg, box = item[0], item[1], item[2] if len(item) > 2 else None
            if box is None:
                continue
            x1, y1, x2, y2 = box
            color = BOX_COLORS.get(acc_type, (0, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            frame = self.put_korean_text(
                frame, msg,
                (x1, max(y1 - 26, 0)),
                font_size=15, color=color, bg_color=(0, 0, 0),
            )

        # 이어폰 박스
        for box in (earphone_boxes or []):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
            frame = self.put_korean_text(
                frame, '이어폰 의심',
                (x1, max(y1 - 26, 0)),
                font_size=15, color=(200, 200, 0), bg_color=(0, 0, 0),
            )
        return frame

    # ── 착용물 경고 (모자·마스크·선글라스) ───────────────────
    def draw_accessory_warning(
        self,
        frame: np.ndarray,
        warnings: list,
    ) -> np.ndarray:
        """감지된 착용물 경고를 화면 중앙에 빨간 박스로 표시한다."""
        if not warnings:
            return frame

        h, w = frame.shape[:2]

        # 반투명 빨간 오버레이
        overlay = frame.copy()
        box_h = 60 + len(warnings) * 46
        y1 = h // 2 - box_h // 2
        y2 = y1 + box_h
        x1, x2 = w // 2 - 260, w // 2 + 260
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 제목
        frame = self.put_korean_text(
            frame, '[ 착용물 감지됨 ]',
            (x1 + 55, y1 + 10),
            font_size=22, color=(255, 255, 255),
        )
        # 항목별 경고 메시지 (warnings 항목이 2개 또는 3개 튜플 모두 처리)
        for i, item in enumerate(warnings):
            msg = item[1]
            frame = self.put_korean_text(
                frame, f'  >> {msg}',
                (x1 + 30, y1 + 48 + i * 46),
                font_size=20, color=(255, 220, 80),
            )
        return frame

    # ── 시험 모드 컨닝 경보 표시 ─────────────────────────────
    def draw_exam_status(
        self,
        frame: np.ndarray,
        cheat_events: list,
        earphone_detected: bool,
        total_events: int,
    ) -> np.ndarray:
        """새로 발생한 컨닝 이벤트를 화면 우하단에 표시한다."""
        h, w = frame.shape[:2]

        # 총 의심 이벤트 카운터 (우상단)
        frame = self.put_korean_text(
            frame, f'의심 이벤트: {total_events}건',
            (w - 220, 32),
            font_size=15, color=(80, 80, 255), bg_color=(0, 0, 0),
        )

        # 이어폰 감지 아이콘
        if earphone_detected:
            frame = self.put_korean_text(
                frame, '🎧 이어폰 의심',
                (w - 170, 54),
                font_size=14, color=(0, 180, 255), bg_color=(0, 0, 0),
            )

        # 이번 프레임에서 새로 감지된 이벤트 경보
        from exam_proctor import CHEAT_LABELS
        for i, ev_type in enumerate(cheat_events[:2]):
            label = CHEAT_LABELS.get(ev_type, ev_type)
            frame = self.put_korean_text(
                frame, f'⚠ {label}',
                (10, h - 140 + i * 36),
                font_size=18, color=(0, 0, 255), bg_color=(0, 0, 0),
            )
        return frame

    # ── 음성 알림 ─────────────────────────────────────────────
    def speak(self, key: str):
        """
        key 에 해당하는 한국어 음성을 백그라운드 스레드로 재생한다.
        쿨다운 내 동일 키는 무시한다.
        """
        now = time.time()
        if (now - self._last_voice_time) < self.COOLDOWN_SEC:
            return
        if not self._mixer_ready:
            return
        if key not in self.VOICE_MSGS:
            return

        self._last_voice_time = now
        self._last_voice_key  = key
        msg = self.VOICE_MSGS[key]

        def _play():
            try:
                from gtts import gTTS
                import pygame
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    tmp_path = f.name
                gTTS(text=msg, lang='ko').save(tmp_path)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                os.remove(tmp_path)
            except Exception as e:
                print(f'[FeedbackHandler] 음성 오류: {e}')

        threading.Thread(target=_play, daemon=True).start()

    # ── 경보 조건 판단 → 음성 자동 선택 ──────────────────────
    def trigger_alerts(
        self,
        score:    int,
        ear_ok:   bool,
        head_ok:  bool,
        gaze_ok:  bool,
    ):
        """
        집중도가 낮거나 특정 지표가 NG 이면 적절한 음성 알림을 발동한다.
        우선순위: EAR > HEAD > GAZE > LOW_SCORE
        """
        if not ear_ok:
            self.speak('EAR_ISSUE')
        elif not head_ok:
            self.speak('HEAD_ISSUE')
        elif not gaze_ok:
            self.speak('GAZE_ISSUE')
        elif score < self.ALERT_SCORE:
            self.speak('LOW_SCORE')
