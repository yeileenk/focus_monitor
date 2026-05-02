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
        'POSTURE_ISSUE' : '어깨가 기울어졌습니다. 자세를 바로잡아 주세요.',
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
        post_msg: str,
        ear_ok:   bool,
        head_ok:  bool,
        gaze_ok:  bool,
        post_ok:  bool,
    ) -> np.ndarray:
        """
        프레임에 집중도 정보를 그린다.
        """
        h, w = frame.shape[:2]

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

        # ── 4개 지표 상태 (우측 상단) ─────────────────────────
        indicators = [
            ('눈', ear_ok),
            ('머리', head_ok),
            ('시선', gaze_ok),
            ('자세', post_ok),
        ]
        for i, (label, ok) in enumerate(indicators):
            dot_color = (0, 210, 80) if ok else (0, 60, 220)
            frame = self.put_korean_text(
                frame,
                f'{"✓" if ok else "✗"} {label}',
                (w - 200 + i * 50, 14),
                font_size=16, color=dot_color,
            )

        # ── 하단 경고 메시지 ──────────────────────────────────
        alerts = []
        if not ear_ok:  alerts.append(ear_msg)
        if not head_ok: alerts.append(head_msg)
        if not gaze_ok: alerts.append(gaze_msg)
        if not post_ok: alerts.append(post_msg)

        for j, msg in enumerate(alerts[:2]):
            frame = self.put_korean_text(
                frame, f'⚠ {msg}',
                (10, h - 90 + j * 36),
                font_size=17,
                color=(20, 80, 255),
                bg_color=(0, 0, 0),
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
        post_ok:  bool,
    ):
        """
        집중도가 낮거나 특정 지표가 NG 이면 적절한 음성 알림을 발동한다.
        우선순위: EAR > HEAD > GAZE > POSTURE > LOW_SCORE
        """
        if not ear_ok:
            self.speak('EAR_ISSUE')
        elif not head_ok:
            self.speak('HEAD_ISSUE')
        elif not gaze_ok:
            self.speak('GAZE_ISSUE')
        elif not post_ok:
            self.speak('POSTURE_ISSUE')
        elif score < self.ALERT_SCORE:
            self.speak('LOW_SCORE')
