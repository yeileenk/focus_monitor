# ============================================================
# timer_dialog.py
# 공부 시간 입력 — 터미널 입력 방식
# 입력 형식: MM:SS  (예: 25:00 = 25분, 90:00 = 90분)
# ============================================================

from typing import Optional


def get_study_duration() -> Optional[int]:
    """
    MM:SS 형식의 공부 시간을 터미널에서 입력받아 총 초(int)를 반환한다.
    취소(빈 입력 or Ctrl+C)하면 None 반환.
    """
    print('\n' + '=' * 40)
    print('  스터디 타이머 — 공부 시간 입력')
    print('  형식: MM:SS  (예: 25:00 = 25분)')
    print('  취소: Enter 키만 누르기 또는 Ctrl+C')
    print('=' * 40)

    while True:
        try:
            raw = input('  공부 시간 입력 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n[INFO] 취소됨.')
            return None

        if not raw:
            return None

        try:
            parts = raw.split(':')
            if len(parts) != 2:
                raise ValueError
            mm, ss = int(parts[0]), int(parts[1])
            if ss < 0 or ss >= 60 or mm < 0:
                raise ValueError
            total = mm * 60 + ss
            if total < 10:
                print('  [오류] 최소 00:10 (10초) 이상 입력하세요.')
                continue
            mins, secs = divmod(total, 60)
            print(f'  {mins}분 {secs}초로 타이머를 시작합니다.\n')
            return total
        except ValueError:
            print('  [오류] MM:SS 형식으로 입력하세요. 예) 25:00  /  01:30  /  90:00')
