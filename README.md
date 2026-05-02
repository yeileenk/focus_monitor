# 집중도 모니터링 AI

웹캠 기반 실시간 집중도 측정 및 세션 리포트 자동화 시스템

---

## 파일 구조

```
focus_monitor/
├── main.py               # 메인 실행 파일
├── pose_detector.py      # MediaPipe 랜드마크 추출
├── ergonomics_rules.py   # EAR / 머리 자세 / 시선 / 자세 분석
├── focus_scorer.py       # 가중 평균 점수 산출 + CSV 로깅
├── feedback_handler.py   # 화면 오버레이 + gTTS 음성 알림
├── report.py             # 세션 리포트 PNG 생성
├── requirements.txt      # 의존성 목록
└── logs/                 # 자동 생성 — CSV + PNG 저장 폴더
```

---

## 설치 방법

```bash
# 1. 가상환경 생성 (선택사항이지만 권장)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux

# 2. 의존성 설치
pip install -r requirements.txt
```

> **Windows 한글 폰트**: `malgun.ttf` 가 자동으로 인식됩니다.  
> **macOS**: `AppleGothic.ttf` 를 사용합니다.  
> **Linux**: `sudo apt install fonts-nanum` 으로 한글 폰트 설치 후 실행하세요.

---

## 실행 방법

```bash
python main.py
```

| 키 | 동작 |
|----|------|
| `q` | 종료 + 세션 리포트 자동 생성 |
| `s` | 중간 요약 콘솔 출력 |

---

## 집중도 판단 기준

| 지표 | 방법 | 임계값 |
|------|------|--------|
| 눈 개방도 (EAR) | Eye Aspect Ratio | < 0.25 × 20프레임 |
| 머리 자세 | solvePnP 각도 추정 | Yaw > ±20°, Pitch > ±18° |
| 시선 방향 | 홍채 랜드마크 비율 | 비율 < 0.35 또는 > 0.65 |
| 자세 (어깨) | 어깨 y좌표 차이 각도 | 기울기 > 12° |

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

## 출력물

- `logs/session_YYYYMMDD_HHMMSS.csv` — 프레임별 집중도 기록
- `logs/report_YYYYMMDD_HHMMSS.png` — 세션 리포트 이미지 (자동 생성)

---

## 활용 기술 스택

- **Python 3.9+**
- **MediaPipe** — FaceMesh (468 랜드마크) + Pose (33 랜드마크)
- **OpenCV** — 웹캠 스트림, 이미지 처리
- **gTTS + pygame** — 한국어 TTS 음성 알림
- **Pillow** — 한글 텍스트 오버레이
- **matplotlib + pandas** — 세션 리포트 시각화
