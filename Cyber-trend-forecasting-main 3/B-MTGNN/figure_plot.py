import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from matplotlib.patches import Patch

try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except Exception:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 파일 경로 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
gap_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')
plot_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(gap_dir, 'normalized_gap.csv')
df = pd.read_csv(csv_path)

df_annual = df[['Pair', '2026-Annual']].copy()
df_annual = df_annual.sort_values('2026-Annual')

# --- Bar plot ---
plt.figure(figsize=(10, 4))
sns.set_style("darkgrid")

colors = ['#4393c3' if v >= 0 else '#d6604d' for v in df_annual['2026-Annual']]
bars = plt.barh(df_annual['Pair'], df_annual['2026-Annual'],
                color=colors, edgecolor='black')
ax = plt.gca()

ax.axvline(x=0, color='gray', linewidth=0.8, linestyle='--')

gap_abs_max = df_annual['2026-Annual'].abs().max()
margin = gap_abs_max * 0.35
ax.set_xlim(-gap_abs_max - margin, gap_abs_max + margin)

ax.bar_label(bars, fmt='%.3f', label_type='edge', padding=5, fontsize=12)

plt.title('Normalized Forecast Gap between Countries (2026)', fontsize=16, fontweight='bold')
plt.xlabel('Normalized Gap  (series_i / base_i  −  series_j / base_j)', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=11)
plt.yticks(fontsize=13)

plt.tight_layout()

out_path = os.path.join(plot_dir, 'normalized_gap_bar.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
out_pdf = os.path.join(plot_dir, 'normalized_gap_bar.pdf')
plt.savefig(out_pdf, bbox_inches='tight', format='pdf')
print(f"Saved: {out_path}")
print(f"Saved: {out_pdf}")

plt.show(block=False)
plt.pause(3)
plt.close()


# ==========================================================
# Grouped bar chart: 국가별 정규화 값 2개 + Gap 값 (DDoS 스타일)
# ==========================================================

summary_path = os.path.join(gap_dir, 'normalized_quarterly_summary.csv')
df_summary = pd.read_csv(summary_path)
value_by_country = dict(zip(df_summary['Country'], df_summary['2026-Annual']))

# "A − B" 형태의 Pair 라벨을 A, B로 분리 (하이픈/민줄표 등 구분자 모두 허용)
def split_pair(label):
    parts = re.split(r'\s*[−–-]\s*', label, maxsplit=1)
    return parts[0].strip(), parts[1].strip()

# 국가별 고정 색상 (밝은색 -> 어두운색), Gap은 항상 가장 어두운 네이비
country_colors = {
    'JPY/USD': '#c6dbef',
    'KRW/USD': '#6baed6',
    'US Dollar Index': '#2171b5',
}
gap_color = '#08306b'

# 옆(y축) 그룹 라벨에 쓸 표시용 국가명 (예시 이미지와 동일한 표기)
row_label_names = {
    'JPY/USD': 'JPY',
    'KRW/USD': 'KRW',
    'US Dollar Index': 'US Trade Weighted Dollar Index',
}

group_labels = []
bar_rows = []  # list of (group_idx, label, value, color)

for _, row in df_annual.iterrows():
    pair_label = row['Pair']
    gap_val = row['2026-Annual']
    name_a, name_b = split_pair(pair_label)
    val_a = value_by_country.get(name_a)
    val_b = value_by_country.get(name_b)
    if val_a is None or val_b is None:
        continue

    # 값이 더 큰 국가를 위쪽, Gap을 가운데, 값이 더 작은 국가를 아래쪽에 배치
    if val_a >= val_b:
        top_name, top_val = name_a, val_a
        bottom_name, bottom_val = name_b, val_b
    else:
        top_name, top_val = name_b, val_b
        bottom_name, bottom_val = name_a, val_a

    group_idx = len(group_labels)
    top_label = row_label_names.get(top_name, top_name)
    bottom_label = row_label_names.get(bottom_name, bottom_name)
    group_labels.append(f"{top_label} - {bottom_label}")
    bar_rows.append((group_idx, top_name, top_val, country_colors.get(top_name, '#9ecae1')))
    bar_rows.append((group_idx, 'Gap', abs(gap_val), gap_color))
    bar_rows.append((group_idx, bottom_name, bottom_val, country_colors.get(bottom_name, '#9ecae1')))

# 그룹당 3개 막대를 위에서 아래로 배치 (그룹 사이 간격 포함)
bar_height = 0.8
n_per_group = 3
y_positions = []
for i, (group_idx, name, val, color) in enumerate(bar_rows):
    slot = i % n_per_group
    y_positions.append(group_idx * (n_per_group + 1) + (n_per_group - 1 - slot))

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style("darkgrid")

for (group_idx, name, val, color), y in zip(bar_rows, y_positions):
    ax.barh(y, val, height=bar_height, color=color, edgecolor='black', zorder=3)
    label_text = f"+{val:.3f}" if name == 'Gap' else f"{val:.3f}"
    ax.text(val + max(v for *_, v, _ in bar_rows) * 0.015, y, label_text,
            va='center', ha='left', fontsize=11)

# 그룹 라벨 (그룹 중앙 y좌표에 표시)
group_centers = [g * (n_per_group + 1) + (n_per_group - 1) / 2 for g in range(len(group_labels))]
ax.set_yticks(group_centers)
ax.set_yticklabels(group_labels, fontsize=12)

ax.set_xticks([])
ax.set_xlabel('')
for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

legend_elements = [
    Patch(facecolor=country_colors['JPY/USD'], edgecolor='black', label='JPY(*N)'),
    Patch(facecolor=country_colors['KRW/USD'], edgecolor='black', label='KRW(*N)'),
    Patch(facecolor=country_colors['US Dollar Index'], edgecolor='black', label='US TWDI(*N)'),
    Patch(facecolor=gap_color, edgecolor='black', label='Gap'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          title='*N: Normalized', title_fontsize=9, framealpha=0.9)

plt.title('Normalized Forecast Gap between Countries (2026)', fontsize=16, fontweight='bold')
plt.tight_layout()

out_path2 = os.path.join(plot_dir, 'normalized_gap_grouped_bar.png')
plt.savefig(out_path2, dpi=300, bbox_inches='tight')
out_pdf2 = os.path.join(plot_dir, 'normalized_gap_grouped_bar.pdf')
plt.savefig(out_pdf2, bbox_inches='tight', format='pdf')
print(f"Saved: {out_path2}")
print(f"Saved: {out_pdf2}")

plt.show(block=False)
plt.pause(3)
plt.close()
