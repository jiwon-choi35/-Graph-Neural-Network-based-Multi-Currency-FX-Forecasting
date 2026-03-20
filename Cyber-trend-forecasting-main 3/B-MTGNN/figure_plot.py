import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

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
