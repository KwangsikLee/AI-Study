import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 경로 설정 (예: 맑은 고딕)
# plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# plt.rcParams['font.family'] = 'NanumGothic'  # 리눅스 (설치 필요)
plt.rcParams['font.family'] = 'AppleGothic'  # macOS

print("\n예제 3: 학생 성적 분석 대시보드")
print("-" * 30)

# 샘플 데이터 생성
np.random.seed(42)
n_students = 100

# 현실적인 데이터 생성
math_scores = np.random.normal(75, 15, n_students)
math_scores = np.clip(math_scores, 0, 100)  # 0-100 범위로 제한

science_scores = 0.7 * math_scores + np.random.normal(15, 8, n_students)
science_scores = np.clip(science_scores, 0, 100)

study_hours = np.random.exponential(3, n_students)
study_hours = np.clip(study_hours, 0.5, 12)

# 데이터프레임 생성
df = pd.DataFrame({
    '수학점수': math_scores,
    '과학점수': science_scores,
    '공부시간': study_hours
})

# 2x2 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('학생 성적 분석 대시보드', fontsize=20, fontweight='bold', y=0.98)

# 수학 점수 히스토그램
axes[0, 0].hist(df['수학점수'], bins=20, color='#FF9999', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('수학 점수 분포', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('점수')
axes[0, 0].set_ylabel('학생 수')
mean_math = df['수학점수'].mean()
axes[0, 0].axvline(mean_math, color='red', linestyle='--', linewidth=2,
                   label=f'평균: {mean_math:.1f}점')
axes[0, 0].legend()

# 수학 vs 과학 점수 산점도
scatter = axes[0, 1].scatter(df['수학점수'], df['과학점수'], 
                            c=df['공부시간'], cmap='viridis', alpha=0.6, s=50)
axes[0, 1].set_title('수학 vs 과학 점수 상관관계', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('수학 점수')
axes[0, 1].set_ylabel('과학 점수')

# 상관계수 계산 및 표시
correlation = df['수학점수'].corr(df['과학점수'])
axes[0, 1].text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                transform=axes[0, 1].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('공부시간 (시간)', rotation=270, labelpad=15)
# 공부시간별 평균 점수 (박스플롯)
studygroups = pd.cut(df['공부시간'], bins=3, labels=['적음(0-2시간)', '보통(2-4시간)', '많음(4시간+)'])
df['공부시간그룹'] = studygroups

box_data = []
group_labels = []
for group in ['적음(0-2시간)', '보통(2-4시간)', '많음(4시간+)']:
    groupmath = df[df['공부시간그룹'] == group]['수학점수'].dropna()
    if len(groupmath) > 0:
        box_data.append(groupmath)
        group_labels.append(group)

if box_data:
    axes[1, 0].boxplot(box_data, labels=group_labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_title('공부시간별 수학 점수 분포', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('수학 점수')
    axes[1, 0].tick_params(axis='x', rotation=45)

# 성적 등급별 파이 차트
def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

df['수학등급'] = df['수학점수'].apply(get_grade)
grade_counts = df['수학등급'].value_counts()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
wedges, texts, autotexts = axes[1, 1].pie(grade_counts.values, 
                                          labels=grade_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors[:len(grade_counts)],
                                          startangle=90)

axes[1, 1].set_title('수학 성적 등급 분포', fontsize=14, fontweight='bold')

# 텍스트 크기 조정
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()
