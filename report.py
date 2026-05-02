# ============================================================
# report.py
# 세션 종료 후 집중도 리포트를 생성하고 PNG 로 저장한다.
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')            # GUI 없이 파일 저장 모드
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def generate_report(history: list[dict], summary: dict, output_dir: str = 'logs'):
    """
    세션 기록(history) 과 요약(summary) 을 받아 시각화 PNG 를 생성한다.

    Parameters:
        history    : focus_scorer.history (리스트)
        summary    : focus_scorer.summary() 반환값
        output_dir : 저장 폴더
    """
    if not history:
        print('[report] 기록 없음 — 리포트 생성 생략')
        return

    scores   = [r['score']   for r in history]
    ear_ok   = [r['ear']     for r in history]
    head_ok  = [r['head']    for r in history]
    gaze_ok  = [r['gaze']    for r in history]
    post_ok  = [r['posture'] for r in history]

    # 시간축: 프레임 인덱스 → 초 (30fps 가정)
    times = [i / 30 for i in range(len(scores))]

    fig = plt.figure(figsize=(14, 10), facecolor='#0F1B35')
    fig.suptitle('집중도 모니터링 — 세션 리포트', fontsize=18,
                 color='white', fontweight='bold', y=0.97)

    grid = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # ── 1. 시간대별 집중도 점수 (상단 전체) ──────────────────
    ax1 = fig.add_subplot(grid[0, :])
    ax1.set_facecolor('#1A2D50')

    # 구간 색칠: 경보(0~40), 주의(40~60), 보통(60~80), 집중(80~100)
    ax1.axhspan( 0, 40, alpha=0.15, color='#EF4444')
    ax1.axhspan(40, 60, alpha=0.12, color='#F97316')
    ax1.axhspan(60, 80, alpha=0.10, color='#EAB308')
    ax1.axhspan(80, 100, alpha=0.10, color='#10B981')

    # 점수 라인
    ax1.plot(times, scores, color='#14B8A6', linewidth=1.8, zorder=3)
    ax1.fill_between(times, scores, alpha=0.18, color='#14B8A6')

    # 이동평균 (30프레임)
    window = min(30, len(scores))
    ma = np.convolve(scores, np.ones(window) / window, mode='same')
    ax1.plot(times, ma, color='#FBBF24', linewidth=2.0, linestyle='--',
             label=f'{window}프레임 이동평균', zorder=4)

    ax1.set_xlim(0, max(times) if times else 1)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('시간 (초)', color='#94A3B8', fontsize=10)
    ax1.set_ylabel('집중도 점수', color='#94A3B8', fontsize=10)
    ax1.tick_params(colors='#94A3B8')
    ax1.spines['bottom'].set_color('#334155')
    ax1.spines['left'].set_color('#334155')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', fontsize=9,
               facecolor='#0F1B35', edgecolor='#334155', labelcolor='white')

    # ── 2. 집중도 분포 (히스토그램) ──────────────────────────
    ax2 = fig.add_subplot(grid[1, 0])
    ax2.set_facecolor('#1A2D50')
    bins   = [0, 40, 60, 80, 100]
    labels = ['경보', '주의', '보통', '집중']
    colors = ['#EF4444', '#F97316', '#EAB308', '#10B981']
    counts = [
        sum(1 for s in scores if s < 40),
        sum(1 for s in scores if 40 <= s < 60),
        sum(1 for s in scores if 60 <= s < 80),
        sum(1 for s in scores if s >= 80),
    ]
    bars = ax2.bar(labels, counts, color=colors, width=0.6, edgecolor='none')
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(cnt), ha='center', va='bottom', color='white', fontsize=9)
    ax2.set_title('집중도 구간 분포', color='white', fontsize=11, pad=8)
    ax2.tick_params(colors='#94A3B8')
    ax2.spines[:].set_visible(False)
    ax2.set_facecolor('#1A2D50')

    # ── 3. 지표별 OK 비율 (파이 차트) ────────────────────────
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.set_facecolor('#1A2D50')
    n = len(history) or 1
    ok_rates = [
        sum(ear_ok) / n * 100,
        sum(head_ok) / n * 100,
        sum(gaze_ok) / n * 100,
        sum(post_ok) / n * 100,
    ]
    ind_labels  = ['눈(EAR)', '머리', '시선', '자세']
    ind_colors  = ['#14B8A6', '#F59E0B', '#8B5CF6', '#10B981']
    ax3.bar(ind_labels, ok_rates, color=ind_colors, width=0.55, edgecolor='none')
    ax3.axhline(70, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.set_ylim(0, 100)
    ax3.set_title('지표별 정상 비율 (%)', color='white', fontsize=11, pad=8)
    ax3.tick_params(colors='#94A3B8')
    ax3.spines[:].set_visible(False)

    # ── 4. 요약 통계 텍스트 박스 ─────────────────────────────
    ax4 = fig.add_subplot(grid[1, 2])
    ax4.set_facecolor('#1A2D50')
    ax4.axis('off')

    stat_lines = [
        ('평균 집중도',  f"{summary.get('avg_score', '-')}점"),
        ('집중 유지율',  f"{summary.get('focus_pct', '-')}%"),
        ('졸음 이벤트', f"{summary.get('drowsy_events', '-')}회"),
        ('머리 이탈',   f"{summary.get('head_out', '-')}회"),
        ('시선 이탈',   f"{summary.get('gaze_out', '-')}회"),
        ('자세 불량',   f"{summary.get('posture_bad', '-')}회"),
    ]
    for i, (label, val) in enumerate(stat_lines):
        y = 0.88 - i * 0.15
        ax4.text(0.05, y, label, transform=ax4.transAxes,
                 color='#94A3B8', fontsize=11)
        ax4.text(0.98, y, val, transform=ax4.transAxes,
                 color='#14B8A6', fontsize=13, fontweight='bold', ha='right')
        ax4.plot([0, 1], [y - 0.04, y - 0.04], color='#334155', linewidth=0.5,
                 transform=ax4.transAxes)
    ax4.set_title('세션 요약', color='white', fontsize=11, pad=8)

    # ── 5. 지표별 시간 흐름 (스택 라인) ─────────────────────
    ax5 = fig.add_subplot(grid[2, :])
    ax5.set_facecolor('#1A2D50')
    def _smooth(arr, w=15):
        k = np.ones(w) / w
        return np.convolve([1 if v else 0 for v in arr], k, mode='same')

    ax5.plot(times, _smooth(ear_ok),  color='#14B8A6', lw=1.3, label='눈(EAR)')
    ax5.plot(times, _smooth(head_ok), color='#F59E0B', lw=1.3, label='머리')
    ax5.plot(times, _smooth(gaze_ok), color='#8B5CF6', lw=1.3, label='시선')
    ax5.plot(times, _smooth(post_ok), color='#10B981', lw=1.3, label='자세')

    ax5.set_xlim(0, max(times) if times else 1)
    ax5.set_ylim(-0.05, 1.15)
    ax5.set_xlabel('시간 (초)', color='#94A3B8', fontsize=10)
    ax5.set_ylabel('정상 비율', color='#94A3B8', fontsize=10)
    ax5.set_title('지표별 시간 흐름 (1=정상, 0=이탈)', color='white', fontsize=11)
    ax5.tick_params(colors='#94A3B8')
    ax5.spines['bottom'].set_color('#334155')
    ax5.spines['left'].set_color('#334155')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.legend(loc='lower right', fontsize=9,
               facecolor='#0F1B35', edgecolor='#334155', labelcolor='white',
               ncol=4)

    # ── 저장 ──────────────────────────────────────────────────
    Path(output_dir).mkdir(exist_ok=True)
    from datetime import datetime
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(output_dir) / f'report_{ts}.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[report] 리포트 저장: {out_path}')
    return str(out_path)
