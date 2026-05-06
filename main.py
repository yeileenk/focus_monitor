# ============================================================
# main.py
# 집중도 모니터링 AI  —  메인 실행 파일
#
# 실행 방법:
#   python main.py              # 일반 집중 모니터링 모드
#   python main.py --study      # 스터디 타이머 모드
#   python main.py --exam       # 시험 감시 모드 (엄격한 임계값 + 컨닝 탐지)
#
# 종료:
#   'q' 키 → 세션 리포트 자동 생성 후 종료
#   's' 키 → 중간 요약 출력
# ============================================================

import argparse
import cv2
import sys

from pose_detector    import PoseDetector
from ergonomics_rules import (
    analyze_ear,
    analyze_head_pose,
    analyze_gaze,
    analyze_earphone,
    get_ear_consec_threshold,
    EXAM_PITCH_THRESHOLD,
    PITCH_THRESHOLD,
)
from focus_scorer     import FocusScorer
from feedback_handler import FeedbackHandler
from report           import generate_report
from device_detector      import DeviceDetector
from exam_proctor         import ExamProctor
from accessory_detector   import AccessoryDetector
from hat_detector         import YoloHatDetector
from evidence_capture     import EvidenceCapture
from evidence_viewer      import generate_evidence_report
from timer_dialog         import get_study_duration
from exam_registration    import ExamRegistration


def main():
    # ── CLI 인자 파싱 ─────────────────────────────────────────
    parser = argparse.ArgumentParser(description='집중도 모니터링 AI')
    parser.add_argument('--exam',  action='store_true', help='시험 감시 모드')
    parser.add_argument('--study', action='store_true', help='스터디 타이머 모드')
    args = parser.parse_args()
    exam_mode  = args.exam
    study_mode = args.study and not args.exam   # exam 우선

    # 스터디 모드: 타이머 시간 입력
    study_duration = None   # 총 초
    study_start    = None
    if study_mode:
        study_duration = get_study_duration()
        if study_duration is None:
            print('[INFO] 취소됨.')
            return
        import time as _time
        study_start = _time.time()
        mm, ss = divmod(study_duration, 60)
        print(f'[INFO] 스터디 타이머: {mm:02d}:{ss:02d} 시작')

    mode_label = '[시험 감시 모드]' if exam_mode else ('[스터디 타이머 모드]' if study_mode else '[집중도 모니터링 모드]')
    print(f'[INFO] {mode_label}로 시작합니다.')

    # ── 웹캠 초기화 ────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[ERROR] 웹캠을 열 수 없습니다. 장치를 확인하세요.')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] 웹캠 해상도: {actual_w} x {actual_h}')

    # ── 모듈 초기화 ────────────────────────────────────────────
    detector        = PoseDetector()
    scorer          = FocusScorer(log_dir='logs')
    feedback        = FeedbackHandler()
    device_detector    = DeviceDetector()
    proctor            = ExamProctor() if exam_mode else None
    hat_detector       = YoloHatDetector() if exam_mode else None
    accessory_detector = AccessoryDetector(hat_detector) if exam_mode else None

    # 세션 ID = CSV 파일명의 타임스탬프 부분
    session_id = scorer.log_path.stem.replace('session_', '')
    evidence   = EvidenceCapture(session_id, log_dir='logs') if exam_mode else None

    ear_consec_threshold = get_ear_consec_threshold(exam_mode)
    pitch_thr            = EXAM_PITCH_THRESHOLD if exam_mode else PITCH_THRESHOLD
    ear_counter          = 0

    win_title    = 'exam_monitor' if exam_mode else 'focus_monitor'
    registration = None
    quit_flag  = [False]
    frame_size = [actual_w, actual_h]   # 콜백에서 참조할 프레임 크기

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if feedback.is_quit_button_clicked(x, y, frame_size[0], frame_size[1]):
                quit_flag[0] = True

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, on_mouse)

    # ── 시험 모드: 5자세 등록 촬영 ────────────────────────────
    if exam_mode:
        registration = ExamRegistration(session_id, log_dir='logs')
        print('[INFO] 시험 등록 촬영을 시작합니다. 위반 물품 없이 5자세를 찍어주세요.')
        reg_ok = registration.run(
            cap, detector, accessory_detector, device_detector, feedback,
            analyze_head_pose, analyze_earphone, win_title=win_title,
        )
        cv2.setMouseCallback(win_title, on_mouse)  # 메인 루프 콜백 복구
        if not reg_ok:
            print('[INFO] 등록 촬영이 취소됐습니다.')
            cap.release()
            cv2.destroyAllWindows()
            detector.release()
            device_detector.release()
            return
        print(f'[INFO] 등록 완료 — {len(registration.ref_photos)}장 저장: {registration.save_dir}')

    print(f'[INFO] 종료: q 키 또는 화면 우하단 [종료] 버튼  |  중간 요약: s')

    # ── 메인 루프 ──────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('[WARN] 프레임 읽기 실패 — 종료합니다.')
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        frame_size[0], frame_size[1] = w, h

        # ── 랜드마크 추출 ──────────────────────────────────────
        frame, face_lms = detector.process(frame)

        # ── 3개 지표 분석 ──────────────────────────────────────
        ear_status,  ear_msg,  ear_val     = analyze_ear(face_lms, exam_mode)
        head_status, head_msg, yaw, pitch  = analyze_head_pose(face_lms, w, h, exam_mode)
        gaze_status, gaze_msg              = analyze_gaze(face_lms, w, exam_mode)

        # ── 이어폰 감지 (시험 모드에서만) ─────────────────────
        earphone_detected = False
        earphone_boxes    = []
        if exam_mode and face_lms is not None:
            earphone_detected, _, earphone_boxes = analyze_earphone(face_lms, frame)

        # ── 착용물 감지 (시험 모드에서만) ─────────────────────
        accessory_warnings: list = []
        if exam_mode and accessory_detector is not None:
            accessory_warnings = accessory_detector.detect(frame, face_lms)

        # ── EAR 연속 카운터 ────────────────────────────────────
        if ear_status == 'EAR_ISSUE':
            ear_counter += 1
        else:
            ear_counter = 0

        face_visible = (face_lms is not None)
        ear_ok  = face_visible and (ear_counter < ear_consec_threshold)
        head_ok = (head_status == 'GOOD')
        gaze_ok = (gaze_status == 'GOOD')

        # 시선이 화면에 고정돼 있으면 고개가 약간 돌아가도 정상 처리.
        # pitch 이탈(고개 숙임)은 스마트폰 조회 패턴이므로 보정하지 않는다.
        if (exam_mode and not head_ok and gaze_ok
                and abs(yaw) <= 30 and abs(pitch) <= pitch_thr):
            head_ok = True
            head_msg = f'시선 고정 보정 (Yaw {yaw:.1f}°)'
            head_status = 'GOOD'

        # ── 전자기기 감지 ──────────────────────────────────────
        dev_boxes, dev_labels, dev_confs = device_detector.detect(frame)

        # ── 컨닝 패턴 감지 (시험 모드에서만) ──────────────────
        cheat_events: list[str] = []
        if exam_mode and proctor is not None:
            cheat_events = proctor.update(
                head_status       = head_status,
                gaze_status       = gaze_status,
                pitch             = pitch,
                device_detected   = bool(dev_boxes),
                earphone_detected = earphone_detected,
                pitch_threshold   = pitch_thr,
            )

        # ── 증거 사진 저장 + 프록터 로깅 (시험 모드) ─────────
        if exam_mode and evidence is not None:
            if bool(dev_boxes):
                clean = feedback.draw_device_boxes(frame.copy(), dev_boxes, dev_labels, dev_confs)
                evidence.try_save(clean, 'device')
            for item in accessory_warnings:
                ev_type = item[0]
                evidence.try_save(frame, ev_type)
                if proctor and proctor.log_event(ev_type, f'{ev_type} 감지'):
                    pass   # 로그 기록됨
            if earphone_detected:
                evidence.try_save(frame, 'earphone')

        # ── 등록 얼굴 비교 (인물 교체 감지) ──────────────────
        if registration and registration.completed:
            if registration.check_identity(frame, face_lms, scorer.frame_cnt):
                if proctor:
                    proctor.log_event('identity_mismatch', '등록 얼굴과 히스토그램 불일치')
                if evidence is not None:
                    evidence.try_save(frame, 'identity_mismatch')

        # ── 점수 산출 ──────────────────────────────────────────
        score = scorer.update(ear_ok, head_ok, gaze_ok,
                              device_detected=bool(dev_boxes))
        level, color = scorer.get_level(score)

        # ── 화면 오버레이 ──────────────────────────────────────
        if not face_visible:
            frame = feedback.put_korean_text(
                frame, '얼굴을 인식할 수 없습니다',
                (w // 2 - 140, h // 2 - 20),
                font_size=26, color=(0, 60, 255), bg_color=(0, 0, 0),
            )
        frame = feedback.draw_overlay(
            frame, score, level, color,
            ear_msg, head_msg, gaze_msg,
            ear_ok, head_ok, gaze_ok,
            exam_mode=exam_mode,
        )
        frame = feedback.draw_device_boxes(frame, dev_boxes, dev_labels, dev_confs)

        if exam_mode and accessory_warnings:
            frame = feedback.draw_accessory_warning(frame, accessory_warnings)

        if exam_mode:
            frame = feedback.draw_accessory_boxes(frame, accessory_warnings, earphone_boxes)

        if exam_mode:
            frame = feedback.draw_exam_status(
                frame, cheat_events, earphone_detected,
                proctor.summary()['total_cheat_events'] if proctor else 0,
            )

        # ── 음성 알림 ──────────────────────────────────────────
        feedback.trigger_alerts(score, ear_ok, head_ok, gaze_ok)

        # ── 최근 30프레임 평균 ─────────────────────────────────
        recent = scorer.recent_avg(30)
        frame  = feedback.put_korean_text(
            frame, f'최근 1초 평균: {recent}점',
            (w - 260, h - 35),
            font_size=16, color=(200, 200, 200),
        )

        # ── 스터디 타이머 처리 ────────────────────────────────
        time_left = None
        if study_mode and study_duration is not None:
            import time as _time
            elapsed   = _time.time() - study_start
            time_left = max(0.0, study_duration - elapsed)
            frame = feedback.draw_study_timer(frame, time_left, study_duration)
            if time_left == 0:
                quit_flag[0] = True   # 타이머 종료 → 자동 세션 마감

        frame = feedback.draw_quit_button(frame)
        cv2.imshow(win_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or quit_flag[0]:
            break
        elif key == ord('s'):
            summ = scorer.summary()
            print('\n[중간 요약]')
            for k, v in summ.items():
                if k != 'log_path':
                    print(f'  {k}: {v}')
            if exam_mode and proctor:
                ex = proctor.summary()
                print(f"  컨닝 이벤트 총계: {ex['total_cheat_events']}건")

    # ── 세션 종료 ──────────────────────────────────────────────
    print('\n[INFO] 세션을 종료합니다...')
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    device_detector.release()

    dev_summary  = device_detector.summary()
    exam_summary = proctor.summary() if proctor else {}
    summary      = scorer.summary()

    if summary:
        summary.update(dev_summary)
        summary.update(exam_summary)

        print('\n===== 세션 최종 요약 =====')
        print(f"  평균 집중도   : {summary['avg_score']}점")
        print(f"  집중 유지율   : {summary['focus_pct']}%")
        print(f"  졸음 이벤트   : {summary['drowsy_events']}회")
        print(f"  머리 이탈     : {summary['head_out']}회")
        print(f"  시선 이탈     : {summary['gaze_out']}회")
        print(f"  [기기] 감지   : {summary['device_events']}회 / {summary['device_seconds']}초")
        if exam_mode:
            print('  ── 컨닝 패턴 감지 결과 ──')
            print(f"  AI 탭 전환    : {summary.get('ai_tab_events', 0)}회")
            print(f"  스마트폰 조회 : {summary.get('phone_look_events', 0)}회")
            print(f"  시선 이탈     : {summary.get('gaze_cheat_events', 0)}회")
            print(f"  이어폰 감지   : {summary.get('earphone_events', 0)}회")
            print(f"  총 의심 이벤트: {summary.get('total_cheat_events', 0)}건")
        print(f"  CSV 저장 경로 : {summary['log_path']}")
        print('==========================')

        report_path = generate_report(
            scorer.history, summary,
            output_dir='logs', exam_mode=exam_mode,
            exam_events=exam_summary.get('event_log', []),
        )
        def open_file(path: str):
            try:
                import subprocess, platform
                os_name = platform.system()
                if os_name == 'Windows':
                    subprocess.Popen(['start', path], shell=True)
                elif os_name == 'Darwin':
                    subprocess.Popen(['open', path])
                else:
                    subprocess.Popen(['xdg-open', path])
            except Exception:
                pass

        if report_path:
            print(f'[INFO] 리포트 이미지: {report_path}')
            open_file(report_path)

        # ── 등록 사진 리포트 (시험 모드) ──────────────────────
        if exam_mode and registration is not None and registration.completed:
            reg_report = registration.generate_registration_report('logs')
            if reg_report:
                print(f'[INFO] 등록 사진 리포트: {reg_report}')
                open_file(reg_report)

        # ── 증거 사진 리포트 (시험 모드) ──────────────────────
        if exam_mode and evidence is not None and evidence.count > 0:
            print(f'[INFO] 증거 사진 {evidence.count}장 저장됨: {evidence.directory}')
            ev_report = generate_evidence_report(evidence.directory, session_id)
            if ev_report:
                open_file(ev_report)
    else:
        print('[INFO] 기록 없음 — 리포트를 생성하지 않습니다.')


if __name__ == '__main__':
    main()
