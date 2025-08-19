import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import seaborn as sns

font_path='/System/Library/Fonts/AppleGothic.ttf'
# fontprop = fm.FontProperties(fname=font_path, size=10)
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

#기준년도,가입자일련번호,시도코드,성별코드,연령대코드(5세단위),신장(5cm단위),체중(5kg단위),허리둘레,시력(좌),시력(우),청력(좌),청력(우),수축기혈압,이완기혈압,식전혈당(공복혈당),총콜레스테롤,트리글리세라이드,HDL콜레스테롤,LDL콜레스테롤,혈색소,요단백,혈청크레아티닌,혈청지오티(AST),혈청지피티(ALT),감마지티피,흡연상태,음주여부,구강검진수검여부,치아우식증유무,결손치 유무,치아마모증유무,제3대구치(사랑니) 이상,치석

def checkData(df):    
    # 4) 이상치 탐지 (Z-score 기준)
    from scipy import stats

    # 수치형 컬럼만 선택
    numeric_df = df.select_dtypes(include=[np.number])
    # 각 컬럼별 z-score 계산
    z_scores = numeric_df.apply(lambda x: stats.zscore(x, nan_policy='omit'))
    print("각 컬럼별 Z-score:")
    print(z_scores.head())

    # 각 수치형 컬럼별 히스토그램 그리기
    columnsInfo = {}
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for idx, col in enumerate(numeric_df.columns):
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.hist(numeric_df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def correlation_matrix(df):
    # 모든 수치형 컬럼의 상관관계 히트맵
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of All Numeric Columns")
    plt.tight_layout()
    plt.show()

def correlation_hitmap(df):
    import seaborn as sns
    df1 = df.copy()    
    df1 = df1.drop(['흡연상태'], axis=1)
    print(df1.head())

    # 상관계수 계산 (숫자형 변수들에 대해서만)
    correlation_matrix = df1.corr(numeric_only=True)
    # 흡연자여부 상관관계만 추출 (자기 자신 제외)
    smoke_corr1 = correlation_matrix[["흡연자여부"]].drop(["흡연자여부", "현재흡연자여부"], errors='ignore')
    # 현재흡연자여부 상관관계만 추출 (자기 자신 제외)
    smoke_corr2 = correlation_matrix[["현재흡연자여부"]].drop(["흡연자여부", "현재흡연자여부"], errors='ignore')

    # 히트맵 2개를 나란히 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    sns.heatmap(smoke_corr1, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=axes[0])
    axes[0].set_title("흡연자여부와의 상관관계")
    sns.heatmap(smoke_corr2, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=axes[1])
    axes[1].set_title("현재흡연자여부와의 상관관계")
    plt.tight_layout()
    plt.show()

# 흡연과 간 기능사아의 상관관계 분석
def collect_data(df):    
    columns = ['연령대코드(5세단위)', '성별코드','혈청지오티(AST)','혈청지피티(ALT)','감마지티피','흡연상태']
    selected_df = df[columns]
    # 결측치가 있는 행은 모두 제거
    selected_df = selected_df.dropna()
    # 흡연상태가 2, 3이면 True, 아니면 False인 새로운 컬럼 추가
    selected_df['흡연자여부'] = selected_df['흡연상태'].apply(lambda x: True if x in [2, 3] else False)
    selected_df['현재흡연자여부'] = selected_df['흡연상태'].apply(lambda x: True if x == 3 else False)

    # 감마지티피 정상치 남성 11~63IU/L, 여성 8~35IU/L 기준으로 정상/비정상 구분
    def gamma_status(row):
        if row['성별코드'] == 1:  # 남성
            return '정상' if 11 <= row['감마지티피'] <= 63 else '비정상'
        elif row['성별코드'] == 2:  # 여성
            return '정상' if 8 <= row['감마지티피'] <= 35 else '비정상'
        else:
            return '비정상'

    selected_df['감마지티피_상태'] = selected_df.apply(gamma_status, axis=1)

    print(selected_df.head())
    return selected_df

def anal_gammagpt(df):
    # 흡연상태별로 연령대별 감마지티피_상태가 '비정상'인 비율 선그래프
    age_col = '연령대코드(5세단위)'
    status_col = '감마지티피_상태'
    smoke_col = '흡연상태'
    # 연령대 코드와 라벨 매핑
    age_labels = {
        1: '0~4세', 2: '5~9세', 3: '10~14세', 4: '15~19세', 5: '20~24세',
        6: '25~29세', 7: '30~34세', 8: '35~39세', 9: '40~44세', 10: '45~49세',
        11: '50~54세', 12: '55~59세', 13: '60~64세', 14: '65~69세', 15: '70~74세',
        16: '75~79세', 17: '80~84세', 18: '85세+'
    }

    fig, ax1 = plt.subplots(figsize=(12,6))
    # 선그래프: 감마지티피 비정상 비율
    for smoke_value, group in df.groupby(smoke_col):
        abnormal_ratio = (
            group.groupby(age_col)[status_col]
            .apply(lambda x: (x == '비정상').mean())
            .reset_index(name='비정상비율')
        )
        # 연령대 코드 -> 라벨로 변환
        abnormal_ratio['연령대라벨'] = abnormal_ratio[age_col].map(age_labels)
        ax1.plot(abnormal_ratio['연령대라벨'], abnormal_ratio['비정상비율'], marker='o', label=f'흡연상태={smoke_value}')

    abnormal_ratio2 = (
            df.groupby(age_col)[status_col]
            .apply(lambda x: (x == '비정상').mean())
            .reset_index(name='비정상비율')
        )
    abnormal_ratio2['연령대라벨'] = abnormal_ratio2[age_col].map(age_labels)
    ax1.plot(abnormal_ratio2['연령대라벨'], abnormal_ratio2['비정상비율'], marker='o', label=f'평균')

    # 막대그래프: 연령대별 현재흡연자 비율
    smoke_ratio = (
        df.groupby(age_col)['현재흡연자여부']
        .mean()
        .reset_index(name='현재흡연자비율')
    )
    smoke_ratio['연령대라벨'] = smoke_ratio[age_col].map(age_labels)
    ax2 = ax1.twinx()
    ax2.bar(smoke_ratio['연령대라벨'], smoke_ratio['현재흡연자비율'], color='gray', alpha=0.3, width=0.7, label='현재흡연자비율')
    ax2.set_ylabel('현재흡연자 비율', color='gray')
    ax2.set_ylim(0, 1)

    ax1.set_title('흡연상태별 연령대별 감마지티피 비정상 비율 & 현재흡연자 비율')
    ax1.set_xlabel('연령대')
    ax1.set_ylabel('비정상 비율')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("./국민건강보험공단_건강검진정보_2023.csv", encoding='utf8')
    # 예시: '가입자일련번호', '시도코드' 컬럼 제거
    df = df.drop(['결손치 유무', '치아마모증유무', '제3대구치(사랑니) 이상', '기준년도'], axis=1)
    
    # print(f"결측치 확인 총 row:{ len(df)}")
    # print(df.isnull().sum())  # 각 컬럼의 결측치 개수 출력

    # checkData(df)
    selecte_df = collect_data(df)

    # 흡연 유무로 분석
    # 현재 흡연유무로 분석

    print(f"결측치 확인 총 row:{ len(selecte_df)}")
    print(selecte_df.isnull().sum())  # 각 컬럼의 결측치 개수 출력

    # correlation_matrix(selecte_df)
    # correlation_hitmap(selecte_df)
    anal_gammagpt(selecte_df)

if __name__ == "__main__":
    main()

