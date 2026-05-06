# ============================================================
# exam_proctor.py
# 시험 감시 모드 — 컨닝 패턴 감지 및 이벤트 기록 모듈
#
# 감지 패턴:
#   1. AI탭전환   — HEAD_ISSUE + GAZE_ISSUE 동시 발생
#   2. 스마트폰조회 — 아래를 향한 고개(Pitch↑) 반복 패턴
#   3. 시선이탈    — GAZE_ISSUE 지속 (옆 화면·메모지)
#   4. 기기감지    — YOLO 폰 탐지
#   5. 이어폰감지  — FaceMesh 귀 영역 분석
# ============================================================

import time
from collections import deque
from typing import Optional


# 패턴별 한글 라벨
CHEAT_LABELS = {
    'ai_tab'            : 'AI 탭 전환 의심',
    'phone_look'        : '스마트폰 조회 의심',
    'gaze_cheat'        : '시선 이탈 (메모·옆 화면)',
    'device'            : '기기 감지',
    'earphone'          : '이어폰 착용 의심',
    'hat'               : '모자 착용 감지',
    'sunglasses'        : '선글라스 착용 감지',
    'mask'              : '마스크 착용 감지',
    'identity_mismatch' : '등록 인물과 불일치 의심',
}

# 연속 GAZE_ISSUE 판정 기준 (초)
GAZE_PERSIST_SEC   = 2.0
# HEAD+GAZE 동시 발생 판정 기준 (초)
AI_TAB_PERSIST_SEC = 1.5
# 아래 고개 반복 감지 — 슬라이딩 윈도우(초) 내 N회 이상
PHONE_LOOK_WINDOW_SEC  = 15.0
PHONE_LOOK_MIN_EVENTS  = 3
# 동일 이벤트 쿨다운 (초) — 중복 기록 방지
COOLDOWN = {
    'ai_tab'            : 8.0,
    'phone_look'        : 12.0,
    'gaze_cheat'        : 6.0,
    'device'            : 5.0,
    'earphone'          : 20.0,
    'hat'               : 15.0,
    'sunglasses'        : 15.0,
    'mask'              : 15.0,
    'identity_mismatch' : 30.0,
}


class ExamProctor:
    """
    매 프레임의 분석 결과를 받아 컨닝 패턴을 감지하고 이벤트 로그를 누적한다.
    """

    def __init__(self):
        self.events: list[dict] = []   # {'time', 'type', 'label', 'detail'}

        # 상태 추적
        self._gaze_issue_start:   Optional[float] = None
        self._ai_tab_start:       Optional[float] = None
        self._downward_times:     deque            = deque()   # Pitch 이탈 타임스탬프

        # 이벤트 타입별 마지막 기록 시각 (쿨다운용)
        self._last_event_time: dict[str, float] = {k: 0.0 for k in COOLDOWN}

    # ── 매 프레임 호출 ────────────────────────────────────────
    def update(
        self,
        head_status: str,
        gaze_status: str,
        pitch:       float,
        device_detected: bool,
        earphone_detected: bool,
        pitch_threshold: float,
    ) -> list[str]:
        """
        분석 결과를 받아 이번 프레임에서 새로 발생한 이벤트 타입 목록을 반환한다.
        """
        now = time.time()
        new_events: list[str] = []

        head_issue = (head_status == 'HEAD_ISSUE')
        gaze_issue = (gaze_status == 'GAZE_ISSUE')
        down_look  = (pitch < -pitch_threshold)   # 아래 고개 (pitch 음수 = 고개 숙임)

        # ── 1. AI 탭 전환: HEAD + GAZE 동시 이탈 ────────────
        if head_issue and gaze_issue:
            if self._ai_tab_start is None:
                self._ai_tab_start = now
            elif now - self._ai_tab_start >= AI_TAB_PERSIST_SEC:
                if self._can_log('ai_tab', now):
                    self._log('ai_tab', now, 'HEAD+GAZE 동시 이탈')
                    new_events.append('ai_tab')
                    self._ai_tab_start = now   # 리셋
        else:
            self._ai_tab_start = None

        # ── 2. 스마트폰 조회: 아래 고개 반복 패턴 ───────────
        if down_look:
            self._downward_times.append(now)
        # 윈도우 밖 항목 제거
        while self._downward_times and now - self._downward_times[0] > PHONE_LOOK_WINDOW_SEC:
            self._downward_times.popleft()
        if len(self._downward_times) >= PHONE_LOOK_MIN_EVENTS:
            if self._can_log('phone_look', now):
                self._log('phone_look', now,
                          f'{PHONE_LOOK_WINDOW_SEC:.0f}초 내 하향 고개 {len(self._downward_times)}회')
                new_events.append('phone_look')
                self._downward_times.clear()

        # ── 3. 시선 이탈 지속 ────────────────────────────────
        if gaze_issue and not head_issue:   # HEAD와 분리된 순수 시선 이탈
            if self._gaze_issue_start is None:
                self._gaze_issue_start = now
            elif now - self._gaze_issue_start >= GAZE_PERSIST_SEC:
                if self._can_log('gaze_cheat', now):
                    self._log('gaze_cheat', now,
                              f'시선 이탈 {now - self._gaze_issue_start:.1f}초 지속')
                    new_events.append('gaze_cheat')
                    self._gaze_issue_start = now
        else:
            self._gaze_issue_start = None

        # ── 4. 기기 감지 ─────────────────────────────────────
        if device_detected and self._can_log('device', now):
            self._log('device', now, 'YOLO 기기 감지')
            new_events.append('device')

        # ── 5. 이어폰 감지 ───────────────────────────────────
        if earphone_detected and self._can_log('earphone', now):
            self._log('earphone', now, '귀 영역 이어폰 의심')
            new_events.append('earphone')

        return new_events

    # ── 외부 이벤트 직접 로깅 (착용물·인물 불일치) ──────────
    def log_event(self, event_type: str, detail: str = '') -> bool:
        """
        update() 외부에서 이벤트를 직접 기록한다.
        쿨다운 내 중복이면 False, 기록되면 True 반환.
        """
        now = time.time()
        if event_type not in self._last_event_time:
            self._last_event_time[event_type] = 0.0
        if not self._can_log(event_type, now):
            return False
        self._log(event_type, now, detail)
        return True

    # ── 내부 헬퍼 ────────────────────────────────────────────
    def _can_log(self, event_type: str, now: float) -> bool:
        return now - self._last_event_time[event_type] >= COOLDOWN[event_type]

    def _log(self, event_type: str, now: float, detail: str):
        self._last_event_time[event_type] = now
        self.events.append({
            'time'  : now,
            'type'  : event_type,
            'label' : CHEAT_LABELS[event_type],
            'detail': detail,
        })

    # ── 세션 요약 ─────────────────────────────────────────────
    def summary(self) -> dict:
        counts = {k: 0 for k in CHEAT_LABELS}
        for e in self.events:
            counts[e['type']] = counts.get(e['type'], 0) + 1
        return {
            'total_cheat_events'    : len(self.events),
            'ai_tab_events'         : counts['ai_tab'],
            'phone_look_events'     : counts['phone_look'],
            'gaze_cheat_events'     : counts['gaze_cheat'],
            'device_events_exam'    : counts['device'],
            'earphone_events'       : counts['earphone'],
            'hat_events'            : counts['hat'],
            'sunglasses_events'     : counts['sunglasses'],
            'mask_events'           : counts['mask'],
            'identity_mismatch_events': counts['identity_mismatch'],
            'event_log'             : self.events,
        }
