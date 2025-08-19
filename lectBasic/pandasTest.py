import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# 소수점 한 자리까지만 표시
pd.options.display.float_format = '{:.1f}'.format

# 초기 성적표 데이터 생성
data = {'이름': ['김철수', '이영희', '박민수'],
        '국어': [90, 85, 92],
        '영어': [88, 92, 80],
        '수학': [95, 80, 97]}
df = pd.DataFrame(data)

# 새로운 학생 성적 추가 (딕셔너리 사용)
new_student = {'이름': '최지혜', '국어': 87, '영어': 95, '수학': 90}
df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True) # Use pd.concat instead of append

# 여러 학생 성적 추가 (리스트 사용)
new_students = [{'이름': '정우진', '국어': 78, '영어': 88, '수학': 92},
                {'이름': '한서연', '국어': 95, '영어': 90, '수학': 85}]
df = pd.concat([df, pd.DataFrame(new_students)], ignore_index=True) # Use pd.concat instead of append

# 최지혜 영어 점수 변경
df.loc[df['이름'] == '최지혜', '영어'] = 100

df['총점1'] = df[['국어', '영어', '수학']].sum(axis=1)  # 총점 계산
df['평균1'] = df[['국어', '영어', '수학']].mean(axis=1).round(1)  # 평균 점수 계산

#과목별 평균
subject_means = df[['국어', '영어', '수학']].mean().round(1)
print("\n과목별 평균 점수:\n", subject_means)

aaa = df['평균1']
print("\n과목별 평균 점수2:\n", aaa)

# 과목별 총점 계산
# df.loc['총점'] = df[['국어', '영어', '수학']].sum()

# 과목별 평균 계산
# df.loc['평균'] = df[['국어', '영어', '수학']].mean() # 잘못된 계산

# subject_means = df.loc['평균', ['국어', '영어', '수학']]
# print("\n과목별 평균 점수:\n", subject_means)

# 평균 점수를 기준으로 오름차순으로 정렬
df_sorted = df.sort_values(by='총점1')

# 정렬된 데이터프레임 출력
# print(df_sorted)


# 과목별 총점 계산
subject_totals = df[['국어', '영어', '수학']].sum()

# 학생 수 계산
num_students = len(df)

# 과목별 평균 점수 계산
subject_means = subject_totals / num_students
# 결과 출력
print("학생수:\n", num_students)
# 결과 출력
print("과목별 평균 점수6:\n", subject_means)


# 결과 출력
print(df)



# 그래프 설정
fig, ax = plt.subplots(figsize=(12, 8))

# 막대 그래프 위치 설정
x = np.arange(len(df['이름']))
width = 0.25

# 막대 그래프 그리기
bars1 = ax.bar(x - width, df['국어'], width, label='국어', color='#ff6b6b', alpha=0.8)
bars2 = ax.bar(x, df['영어'], width, label='영어', color='#4ecdc4', alpha=0.8)
bars3 = ax.bar(x + width, df['수학'], width, label='수학', color='#45b7d1', alpha=0.8)

# 그래프 꾸미기
ax.set_xlabel('학생', fontsize=12)
ax.set_ylabel('점수', fontsize=12)
ax.set_title('학생별 과목 성적', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['이름'])
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# 막대 위에 점수 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 과목별 점수 데이터 확인
print("학생별 과목 점수:")
print(df)