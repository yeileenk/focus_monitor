# 집중도 모니터링 AI

웹캠 기반 실시간 집중도 측정 및 비대면 시험 감시 시스템

---

## 파일 구조

```
focus_monitor/
├── main.py                # 메인 실행 파일 (일반 / 시험 모드 분기)
├── pose_detector.py       # MediaPipe FaceMesh + Pose 랜드마크 추출
├── ergonomics_rules.py    # EAR / 머리 자세 / 시선 / 자세 분석 (모드별 임계값)
├── focus_scorer.py        # 가중 평균 점수 산출 + CSV 로깅
├── feedback_handler.py    # 화면 오버레이 + 음성 알림 + 경고 UI
├── report.py              # 세션 리포트 PNG 생성
├── device_detector.py     # YOLOv8 전자기기 실시간 감지
├── exam_proctor.py        # 시험 모드 컨닝 패턴 감지 및 이벤트 기록
├── accessory_detector.py  # 모자 / 마스크 / 선글라스 감지
├── evidence_capture.py    # 부정행위 의심 시 증거 사진 자동 저장
├── evidence_viewer.py     # 세션 종료 후 증거 사진 그리드 리포트 생성
├── requirements.txt       # 의존성 목록
└── logs/                  # 자동 생성
    ├── session_*.csv          # 프레임별 집중도 기록
    ├── report_*.png           # 세션 리포트 이미지
    └── evidence_*/            # 부정행위 의심 증거 사진 (시험 모드)
        └── evidence_report_*.png  # 증거 사진 그리드 리포트
```

---

## 설치 방법

```bash
# 1. 가상환경 생성 및 활성화 (권장)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# 2. 의존성 설치
pip install -r requirements.txt
```

> **macOS**: `AppleGothic.ttf` 를 자동으로 인식합니다.  
> **Windows**: `malgun.ttf` 가 자동으로 인식됩니다.  
> **Linux**: `sudo apt install fonts-nanum` 으로 한글 폰트 설치 후 실행하세요.

---

## 실행 방법

### 일반 집중 모니터링 모드

```bash
python main.py
```

### 시험 감시 모드

```bash
python main.py --exam
```

| 키 / 버튼 | 동작 |
|-----------|------|
| `q` | 종료 + 세션 리포트 자동 생성 |
| `s` | 중간 요약 콘솔 출력 |
| 화면 우하단 `종료` 버튼 | 마우스 클릭으로 종료 |

---

## 집중도 판단 기준

| 지표 | 방법 | 일반 모드 임계값 | 시험 모드 임계값 |
|------|------|-----------------|-----------------|
| 눈 개방도 (EAR) | Eye Aspect Ratio | < 0.25 × 20프레임 | < 0.25 × 10프레임 |
| 머리 자세 | solvePnP 각도 추정 | Yaw > ±30°, Pitch > ±25° | Yaw > ±12°, Pitch > ±15° |
| 시선 방향 | 홍채 랜드마크 비율 | 비율 < 0.35 또는 > 0.65 | 비율 < 0.38 또는 > 0.62 |
| 자세 (어깨) | 어깨 y좌표 차이 각도 | 기울기 > 12° | 기울기 > 12° |

> 얼굴이 인식되지 않으면 (얼굴 가림 등) EAR·머리·시선 전부 실패로 처리됩니다.

### 점수 가중치

| 지표 | 가중치 |
|------|--------|
| 눈 개방도 | 35% |
| 머리 자세 | 30% |
| 시선 방향 | 25% |
| 자세 | 10% |

### 점수 구간

| 점수 | 상태 |
|------|------|
| 80 ~ 100 | 집중 중 |
| 60 ~ 79  | 보통 |
| 40 ~ 59  | 주의 |
| 0  ~ 39  | 경보 |

---

## 시험 감시 모드 기능

### 컨닝 패턴 자동 감지

| 패턴 | 감지 방법 | 판정 기준 |
|------|-----------|-----------|
| AI 탭 전환 의심 | HEAD_ISSUE + GAZE_ISSUE 동시 발생 | 1.5초 이상 지속 |
| 스마트폰 조회 의심 | 아래 고개(Pitch↑) 반복 | 15초 내 3회 이상 |
| 시선 이탈 (메모·옆 화면) | GAZE_ISSUE 단독 지속 | 2초 이상 지속 |
| 전자기기 감지 | YOLOv8 객체 감지 | cell phone / laptop / remote 등 |
| 이어폰 착용 의심 | FaceMesh 귀 영역 픽셀 분석 | 귀 주변 분산 + 밝기 기준 |

### 금지 착용물 감지

| 착용물 | 감지 방법 | 비고 |
|--------|-----------|------|
| 모자 | 이마 위 ROI 밝기 + 균일도 분석 | 어둡고 균일한 영역 = 모자 |
| 마스크 | 코~턱 영역 색상 균일도 분석 | 피부 대비 균일하고 색 다름 |
| 선글라스 | 눈 영역 밝기 분석 | 피부 밝기의 65% 미만 |
| 일반 안경 | — | 오탐 없음 (투명 렌즈) |

감지 시 화면 중앙에 빨간 경고창 표시 → 착용물 제거 시 자동 해제.

---

## 전자기기 감지

YOLOv8n 모델(COCO 데이터셋)을 사용해 웹캠 화면에서 실시간으로 전자기기를 탐지합니다.

**감지 대상**: `cell phone`, `laptop`, `remote`, `keyboard`, `mouse`

- 감지 시 화면에 빨간 바운딩 박스 + 라벨 표시
- 세션 요약에 감지 횟수 / 총 사용 시간 기록
- 첫 실행 시 YOLOv8n 모델 자동 다운로드 (~6MB)

---

## 출력물

| 파일 | 설명 |
|------|------|
| `logs/session_*.csv` | 프레임별 집중도 기록 |
| `logs/report_*.png` | 세션 리포트 (점수 그래프, 지표별 분포, 요약 통계) |
| `logs/evidence_*/` | 부정행위 의심 증거 사진 (시험 모드) |
| `logs/evidence_*/evidence_report_*.png` | 증거 사진 그리드 리포트 (시험 모드) |

세션 종료 시 리포트와 증거 사진 리포트가 자동으로 열립니다.

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.9+ |
| 얼굴·자세 감지 | MediaPipe FaceMesh (468 랜드마크) + Pose (33 랜드마크) |
| 객체 감지 | YOLOv8n (ultralytics) |
| 영상 처리 | OpenCV |
| 음성 알림 | gTTS + pygame |
| 텍스트 오버레이 | Pillow (PIL) |
| 시각화 | matplotlib |
| 데이터 처리 | pandas, numpy, scipy |
