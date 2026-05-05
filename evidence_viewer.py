# ============================================================
# evidence_viewer.py
# 세션 종료 후 증거 사진을 한 페이지에 모아서 PNG로 저장·열기
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

LABEL_KO = {
    'device'    : '전자기기',
    'hat'       : '모자',
    'mask'      : '마스크',
    'sunglasses': '선글라스',
    'earphone'  : '이어폰',
}

TYPE_COLORS = {
    'device'    : '#FBBF24',
    'hat'       : '#F97316',
    'mask'      : '#EF4444',
    'sunglasses': '#8B5CF6',
    'earphone'  : '#06B6D4',
}


def generate_evidence_report(evidence_dir: str, session_id: str):
    """
    evidence_dir 안의 JPG 파일을 그리드로 배치해 PNG 리포트를 생성한다.

    Returns:
        저장된 PNG 경로 (없으면 None)
    """
    ev_path = Path(evidence_dir)
    images  = sorted(ev_path.glob('*.jpg'))

    if not images:
        print('[evidence] 저장된 증거 사진이 없습니다.')
        return None

    n    = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig_w = cols * 4.2
    fig_h = rows * 3.4 + 1.2

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), facecolor='#0F1B35')
    fig.suptitle(f'부정행위 의심 증거 사진  —  세션 {session_id}  ({n}장)',
                 fontsize=15, color='white', fontweight='bold', y=0.99)

    # axes를 항상 2D 리스트로 통일
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    for idx, img_path in enumerate(images):
        r, c = divmod(idx, cols)
        ax   = axes[r][c]

        # 이미지 로드
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis('off')

        # 파일명에서 타입 파싱 (예: 003_142305_device.jpg)
        stem   = img_path.stem                      # 003_142305_device
        parts  = stem.split('_')
        ev_type = parts[-1] if parts else 'unknown'
        label   = LABEL_KO.get(ev_type, ev_type)
        color   = TYPE_COLORS.get(ev_type, '#FFFFFF')
        ts      = parts[1] if len(parts) >= 2 else ''
        ts_fmt  = f'{ts[:2]}:{ts[2:4]}:{ts[4:6]}' if len(ts) >= 6 else ts

        ax.set_title(f'[{ts_fmt}]  {label}',
                     color=color, fontsize=9, pad=4, fontweight='bold')

    # 빈 칸 숨기기
    total_cells = rows * cols
    for idx in range(n, total_cells):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')
        axes[r][c].set_facecolor('#0F1B35')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = ev_path / f'evidence_report_{session_id}.png'
    fig.savefig(str(out_path), dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[evidence] 증거 리포트 저장: {out_path}')
    return str(out_path)
