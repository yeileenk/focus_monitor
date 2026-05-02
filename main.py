# ============================================================
# main.py
# 집중도 모니터링 AI  —  메인 실행 파일
#
# 실행 방법:
#   python main.py
#
# 종료:
#   'q' 키 → 세션 리포트 자동 생성 후 종료
#   's' 키 → 중간에 리포트 미리 보기
#
# 파일 구조:
#   main.py              ← 지금 이 파일
#   pose_detector.py     ← MediaPipe 랜드마크 추출
#   ergonomics_rules.py  ← EAR / 머리 자세 / 시선 / 자세 분석
#   focus_scorer.py      ← 가중 평균 점수 산출 + CSV 로깅
#   feedback_handler.py  ← 화면 오버레이 + gTTS 음성 알림
#   report.py            ← 세션 리포트 PNG 생성
# ============================================================

import cv2
import sys

from pose_detector    import PoseDetector
from ergonomics_rules import (
    analyze_ear,
    analyze_head_pose,
    analyze_gaze,
    analyze_posture,
    EAR_CONSEC_FRAMES,
)
from focus_scorer    import FocusScorer
from feedback_handler import FeedbackHandler
from report           import generate_report


def main():
    # ── 웹캠 초기화 ────────────────────────────────────────────
    cap = cv2.VideoCapture(0)         # 0 = 기본 웹캠
    if not cap.isOpened():
        print('[ERROR] 웹캠을 열 수 없습니다. 장치를 확인하세요.')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] 웹캠 해상도: {actual_w} x {actual_h}')

    # ── 모듈 초기화 ────────────────────────────────────────────
    detector = PoseDetector()
    scorer   = FocusScorer(log_dir='logs')
    feedback = FeedbackHandler()

    # EAR 연속 프레임 카운터 (졸음 판단용)
    ear_counter = 0

    print('[INFO] 집중도 모니터링을 시작합니다. 종료하려면 q 를 누르세요.')
    print('[INFO] 세션 리포트 미리보기: s 키')

    # ── 메인 루프 ──────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('[WARN] 프레임 읽기 실패 — 종료합니다.')
            break

        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── 랜드마크 추출 ──────────────────────────────────────
        frame, face_lms, pose_lms = detector.process(frame)

        # ── 4개 지표 분석 ──────────────────────────────────────
        ear_status,  ear_msg,  ear_val              = analyze_ear(face_lms)
        head_status, head_msg, yaw, pitch           = analyze_head_pose(face_lms, w, h)
        gaze_status, gaze_msg                       = analyze_gaze(face_lms, w)
        post_status, post_msg                       = analyze_posture(pose_lms)

        # ── EAR 연속 프레임 카운터 ─────────────────────────────
        if ear_status == 'EAR_ISSUE':
            ear_counter += 1
        else:
            ear_counter = 0

        # 연속 N 프레임 미만이면 일시적 깜빡임 → GOOD 으로 처리
        ear_ok  = (ear_counter < EAR_CONSEC_FRAMES)
        head_ok = (head_status in ('GOOD', 'NO_FACE'))
        gaze_ok = (gaze_status in ('GOOD', 'NO_FACE'))
        post_ok = (post_status == 'GOOD')

        # ── 점수 산출 ──────────────────────────────────────────
        score = scorer.update(ear_ok, head_ok, gaze_ok, post_ok)
        level, color = scorer.get_level(score)

        # ── 화면 오버레이 ──────────────────────────────────────
        frame = feedback.draw_overlay(
            frame, score, level, color,
            ear_msg, head_msg, gaze_msg, post_msg,
            ear_ok, head_ok, gaze_ok, post_ok,
        )

        # ── 음성 알림 ──────────────────────────────────────────
        feedback.trigger_alerts(score, ear_ok, head_ok, gaze_ok, post_ok)

        # ── 최근 30프레임 평균 표시 ────────────────────────────
        recent = scorer.recent_avg(30)
        frame  = feedback.put_korean_text(
            frame, f'최근 1초 평균: {recent}점',
            (w - 260, h - 35),
            font_size=16, color=(200, 200, 200),
        )

        # ── 프레임 출력 ────────────────────────────────────────
        cv2.imshow('집중도 모니터링 AI', frame)

        # ── 키 입력 처리 ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 중간 리포트 미리보기
            summ = scorer.summary()
            print('\n[중간 요약]')
            for k, v in summ.items():
                print(f'  {k}: {v}')

    # ── 세션 종료 처리 ─────────────────────────────────────────
    print('\n[INFO] 세션을 종료합니다...')
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    summary = scorer.summary()
    if summary:
        print('\n===== 세션 최종 요약 =====')
        print(f"  평균 집중도   : {summary['avg_score']}점")
        print(f"  집중 유지율   : {summary['focus_pct']}%")
        print(f"  졸음 이벤트   : {summary['drowsy_events']}회")
        print(f"  머리 이탈     : {summary['head_out']}회")
        print(f"  시선 이탈     : {summary['gaze_out']}회")
        print(f"  자세 불량     : {summary['posture_bad']}회")
        print(f"  CSV 저장 경로 : {summary['log_path']}")
        print('==========================')

        # 리포트 PNG 자동 생성
        report_path = generate_report(scorer.history, summary, output_dir='logs')
        if report_path:
            print(f'[INFO] 리포트 이미지: {report_path}')

            # 리포트 이미지 자동으로 열기 (선택사항)
            try:
                import subprocess, platform
                os_name = platform.system()
                if os_name == 'Windows':
                    subprocess.Popen(['start', report_path], shell=True)
                elif os_name == 'Darwin':
                    subprocess.Popen(['open', report_path])
                else:
                    subprocess.Popen(['xdg-open', report_path])
            except Exception:
                pass
    else:
        print('[INFO] 기록 없음 — 리포트를 생성하지 않습니다.')


if __name__ == '__main__':
    main()
