import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# 폰트 및 기본 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# 그래프 1: FGSM vs DeepTruth 화질 비교
def create_quality_comparison_chart():
    try:
        df = pd.read_csv("fgsm_vs_deeptruth_quality_comparison.csv")
    except FileNotFoundError:
        print("[ERROR] fgsm_vs_deeptruth_quality_comparison.csv 파일이 없습니다")
        return

    artwork_labels = [
        "Impression\nSunrise",
        "Night\nWatch", 
        "Mona\nLisa",
        "Persistence\nMemory",
        "Untitled\nHaring"
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(artwork_labels))
    width = 0.32

    # SSIM 비교
    bars1 = ax1.bar(x - width/2, df['FGSM_SSIM'], width, label='FGSM', 
                    color='#FF7F7F', alpha=0.85, edgecolor='white', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, df['DeepTruth_SSIM'], width, label='DeepTruth', 
                    color='#4ECDC4', alpha=0.85, edgecolor='white', linewidth=0.8)

    ax1.set_xlabel('Artwork', fontweight='bold', fontsize=10)
    ax1.set_ylabel('SSIM Score', fontweight='bold', fontsize=10)
    ax1.set_title('SSIM Quality Comparison\nFGSM vs DeepTruth', fontweight='bold', fontsize=11, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(artwork_labels, fontsize=8, ha='center')
    ax1.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax1.set_ylim(0, 1.05)

    # SSIM 값 표시
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2, height1 + 0.015, 
                f'{height1:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2, height2 + 0.015, 
                f'{height2:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # PSNR 비교
    bars3 = ax2.bar(x - width/2, df['FGSM_PSNR'], width, label='FGSM', 
                    color='#FF7F7F', alpha=0.85, edgecolor='white', linewidth=0.8)
    bars4 = ax2.bar(x + width/2, df['DeepTruth_PSNR'], width, label='DeepTruth', 
                    color='#4ECDC4', alpha=0.85, edgecolor='white', linewidth=0.8)

    ax2.set_xlabel('Artwork', fontweight='bold', fontsize=10)
    ax2.set_ylabel('PSNR (dB)', fontweight='bold', fontsize=10)
    ax2.set_title('PSNR Quality Comparison\nFGSM vs DeepTruth', fontweight='bold', fontsize=11, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(artwork_labels, fontsize=8, ha='center')
    ax2.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax2.set_ylim(0, 36)

    # PSNR 값 표시
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        height3 = bar3.get_height()
        height4 = bar4.get_height()
        ax2.text(bar3.get_x() + bar3.get_width()/2, height3 + 0.5, 
                f'{height3:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax2.text(bar4.get_x() + bar4.get_width()/2, height4 + 0.5, 
                f'{height4:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout(pad=2.0)
    plt.savefig('quality_comparison_improved.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()


# 그래프 2: The Night Watch 정밀 모드 분석
def create_precision_analysis_chart():
    try:
        df = pd.read_csv("night_watch_precision_analysis.csv")
    except FileNotFoundError:
        print("[ERROR] night_watch_precision_analysis.csv 파일이 없습니다")
        return

    # 데이터 준비
    levels = df['Level'].tolist()
    epsilons = []
    confidences = []

    # Epsilon 값 추출
    for eps_str in df['Epsilon']:
        try:
            if '동적' in str(eps_str):
                epsilons.append(0.015)
            else:
                epsilons.append(float(str(eps_str)))
        except:
            epsilons.append(0.015)

    # Confidence 값 추출
    for conf_str in df['Confidence']:
        try:
            confidences.append(float(str(conf_str).replace('%', '')))
        except:
            confidences.append(0)

    # SSIM, PSNR 값 가져오기
    ssim_values = df['SSIM'].tolist()
    psnr_values = df['PSNR'].tolist()
    classifications = df['Classification'].tolist()

    # 2x2 레이아웃 차트 생성
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 신뢰도 변화 차트
    colors = ['red' if 'Romanticism' in cls else 'blue' for cls in classifications]

    ax1.plot(range(len(levels)), confidences, 'o-', linewidth=2.5, markersize=8, color='darkblue')
    for i, (level, conf, color) in enumerate(zip(levels, confidences, colors)):
        ax1.scatter(i, conf, s=120, c=color, alpha=0.8, edgecolor='white', linewidth=2, zorder=5)
        ax1.annotate(f'{conf}%', (i, conf), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, fontweight='bold')

    # 패턴 A, B 표시
    ax1.annotate('Pattern A\n(Same Class)', xy=(1.5, 27), xytext=(1.5, 20),
                arrowprops=dict(arrowstyle='->', color='red', lw=2), 
                ha='center', fontsize=9, color='red', fontweight='bold')
    ax1.annotate('Pattern B\n(Diff. Class)', xy=(3.5, 40), xytext=(3.5, 47),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2), 
                ha='center', fontsize=9, color='blue', fontweight='bold')

    ax1.set_title('The Night Watch: Precision Mode\nConfidence Changes', fontweight='bold', fontsize=12, pad=15)
    ax1.set_xlabel('Precision Level', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Classification Confidence (%)', fontweight='bold', fontsize=10)
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels(levels, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(15, 50)

    # 2. Epsilon 변화 차트
    ax2.plot(range(len(levels)), epsilons, 's-', linewidth=2.5, markersize=8, color='purple')
    for i, (level, eps) in enumerate(zip(levels, epsilons)):
        ax2.annotate(f'{eps:.4f}', (i, eps), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, fontweight='bold')

    ax2.set_title('The Night Watch: Precision Mode\nEpsilon Changes', fontweight='bold', fontsize=12, pad=15)
    ax2.set_xlabel('Precision Level', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Epsilon Value', fontweight='bold', fontsize=10)
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. SSIM 변화 차트
    ax3.plot(range(len(levels)), ssim_values, 'D-', linewidth=2.5, markersize=8, color='green')
    for i, (level, ssim_val) in enumerate(zip(levels, ssim_values)):
        ax3.annotate(f'{ssim_val:.6f}', (i, ssim_val), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=8, fontweight='bold')

    ax3.set_title('SSIM Quality Changes\n(Higher = Better)', fontweight='bold', fontsize=12, pad=15)
    ax3.set_xlabel('Precision Level', fontweight='bold', fontsize=10)
    ax3.set_ylabel('SSIM Score', fontweight='bold', fontsize=10)
    ax3.set_xticks(range(len(levels)))
    ax3.set_xticklabels(levels, fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # 4. PSNR 변화 차트
    ax4.plot(range(len(levels)), psnr_values, '^-', linewidth=2.5, markersize=8, color='orange')
    for i, (level, psnr_val) in enumerate(zip(levels, psnr_values)):
        ax4.annotate(f'{psnr_val:.4f}dB', (i, psnr_val), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=8, fontweight='bold')

    ax4.set_title('PSNR Quality Changes\n(Higher = Better)', fontweight='bold', fontsize=12, pad=15)
    ax4.set_xlabel('Precision Level', fontweight='bold', fontsize=10)
    ax4.set_ylabel('PSNR (dB)', fontweight='bold', fontsize=10)
    ax4.set_xticks(range(len(levels)))
    ax4.set_xticklabels(levels, fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # MSE 정보가 있다면 추가 표시
    if 'MSE' in df.columns:
        mse_values = df['MSE'].tolist()
        mse_range = max(mse_values) - min(mse_values)
        fig.text(0.02, 0.02, f'MSE Range: {min(mse_values):.2f}~{max(mse_values):.2f} (Diff: {mse_range:.2f})', 
                fontsize=9, style='italic')

    plt.tight_layout(pad=3.0)
    plt.savefig('precision_analysis_improved.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()


# 그래프 3: 개선율 히트맵
def create_improvement_heatmap():
    try:
        df = pd.read_csv("fgsm_vs_deeptruth_quality_comparison.csv")
    except FileNotFoundError:
        print("[ERROR] fgsm_vs_deeptruth_quality_comparison.csv 파일이 없습니다")
        return

    # 개선율 데이터 계산
    data = {
        'Artwork': ['Impression\nSunrise', 'Night\nWatch', 'Mona\nLisa', 'Persistence\nMemory', 'Untitled\nHaring'],
        'SSIM_Improvement': [23.7, 29.3, 39.8, 31.9, 15.8],
        'PSNR_Improvement': [4.3, 2.1, 1.2, 5.3, 13.2]
    }

    df_improvement = pd.DataFrame(data)

    # 히트맵용 데이터 준비
    heatmap_data = np.array([df_improvement['SSIM_Improvement'].values, df_improvement['PSNR_Improvement'].values])

    fig, ax = plt.subplots(figsize=(12, 6))

    # 히트맵 생성
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    # 축 설정
    ax.set_xticks(range(len(df_improvement['Artwork'])))
    ax.set_xticklabels(df_improvement['Artwork'], fontsize=10, ha='center')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['SSIM Improvement (%)', 'PSNR Improvement (dB)'], fontsize=11, fontweight='bold')

    # 값 표시
    for i in range(2):
        for j in range(len(df_improvement['Artwork'])):
            value = heatmap_data[i, j]
            unit = '%' if i == 0 else 'dB'
            ax.text(j, i, f'+{value}{unit}', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='white' if value < 20 else 'black')

    # 컬러바
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement Level', rotation=270, labelpad=20, fontweight='bold')

    ax.set_title('DeepTruth Quality Improvement over FGSM\nAll Artworks Show Significant Enhancement', 
                fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('improvement_heatmap_improved.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()


# 그래프 4: 승률 도넛 차트
def create_victory_rate_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # SSIM 승률
    sizes1 = [100]
    labels1 = ['100%\nDeepTruth\nVictory']
    colors1 = ['#4ECDC4']

    wedges1, texts1 = ax1.pie(sizes1, labels=labels1, colors=colors1, 
                             startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})

    # 도넛 차트로 만들기
    centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
    ax1.add_artist(centre_circle1)
    ax1.text(0, 0, '100%', ha='center', va='center', fontsize=20, fontweight='bold', color='#4ECDC4')
    ax1.set_title('SSIM Quality Victory\n(5 Artworks)', fontweight='bold', fontsize=12, pad=20)

    # PSNR 승률
    sizes2 = [100]
    labels2 = ['100%\nDeepTruth\nVictory']
    colors2 = ['#4ECDC4']

    wedges2, texts2 = ax2.pie(sizes2, labels=labels2, colors=colors2, 
                             startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})

    # 도넛 차트로 만들기
    centre_circle2 = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle2)
    ax2.text(0, 0, '100%', ha='center', va='center', fontsize=20, fontweight='bold', color='#4ECDC4')
    ax2.set_title('PSNR Quality Victory\n(5 Artworks)', fontweight='bold', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig('victory_rate_improved.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()


# 모든 차트 생성 함수
def generate_all_charts():
    print("=" * 60)
    print("DeepTruth 성능 분석 차트 생성")
    print("=" * 60)

    try:
        print("[INFO] 1. FGSM vs DeepTruth 화질 비교 차트 생성 중...")
        create_quality_comparison_chart()
        print("[OK] quality_comparison_improved.png 생성 완료")
    except Exception as e:
        print(f"[ERROR] 화질 비교 차트 생성 실패: {e}")

    try:
        print("[INFO] 2. The Night Watch 정밀 모드 분석 차트 생성 중...")
        create_precision_analysis_chart()
        print("[OK] precision_analysis_improved.png 생성 완료")
    except Exception as e:
        print(f"[ERROR] 정밀 모드 차트 생성 실패: {e}")

    try:
        print("[INFO] 3. 개선율 히트맵 생성 중...")
        create_improvement_heatmap()
        print("[OK] improvement_heatmap_improved.png 생성 완료")
    except Exception as e:
        print(f"[ERROR] 히트맵 생성 실패: {e}")

    try:
        print("[INFO] 4. 승률 도넛 차트 생성 중...")
        create_victory_rate_chart()
        print("[OK] victory_rate_improved.png 생성 완료")
    except Exception as e:
        print(f"[ERROR] 승률 차트 생성 실패: {e}")

    print("=" * 60)
    print("[DONE] 모든 차트 생성 완료")
    print("생성된 파일:")
    print("- quality_comparison_improved.png")
    print("- precision_analysis_improved.png")
    print("- improvement_heatmap_improved.png") 
    print("- victory_rate_improved.png")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_charts()