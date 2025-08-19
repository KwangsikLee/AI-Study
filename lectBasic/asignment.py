import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'AppleGothic'  # macOS

def grade(score):
    """점수에 따라 학점을 반환하는 함수"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

students = ['김철수', '이영희', '박민수', '최지은', '정다은', 
           '강호준', '윤서연', '임태현', '조미래', '한예린']

# 과목
subjects = ['국어', '영어', '수학', '과학']

# 성적 데이터 생성 (50-100점 사이)
np.random.seed(42)  # 동일한 결과를 위한 시드
data = {}

for subject in subjects:
    # 각 과목마다 50-100점 사이의 랜덤 점수 생성
    scores = np.random.randint(50, 101, size=10)
    data[subject] = scores

# 데이터프레임 생성
df = pd.DataFrame(data, index=students)

# 평균 계산
df['평균'] = df[subjects].mean(axis=1).round(1)

# 등수 계산
df['등수'] = df['평균'].rank(method='min', ascending=False).astype(int)

# 등급
df['학점'] = df['평균'].apply(grade)

print("\n학생 성적 데이터프레임:")
print(df)

fig, axexs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('학생 성적 분석 대시보드', fontsize=16, fontweight='bold', y=0.98)

# 국어 점수 히스토그램
axexs[0, 0].hist(df['국어'], bins=10, color='#FF9999', alpha=0.7, edgecolor='black')
axexs[0, 0].set_title('국어 점수 분포', fontsize=14, fontweight='bold')
axexs[0, 0].set_xlabel('점수')
axexs[0, 0].set_ylabel('학생 수')
mean_korean = df['국어'].mean()
axexs[0, 0].axvline(mean_korean, color='red', linestyle='--', linewidth=2,
                   label=f'평균: {mean_korean:.1f}점')
axexs[0, 0].legend()       


axexs[0, 1].scatter(df['국어'], df['영어'], 
                    c=df['평균'], cmap='viridis', alpha=0.6, s=100)
axexs[0, 1].set_title('국어 vs 영어 점수 상관관계', fontsize=14, fontweight='bold')
axexs[0, 1].set_xlabel('국어 점수')
axexs[0, 1].set_ylabel('영어 점수')
correlation = df['국어'].corr(df['영어'])
axexs[0, 1].text(0.05, 0.95, f'상관계수: {correlation:.3f}')

plt.tight_layout()
plt.show()