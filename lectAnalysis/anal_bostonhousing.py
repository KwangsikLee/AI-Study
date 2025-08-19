import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt


font_path='/System/Library/Fonts/AppleGothic.ttf'
# fontprop = fm.FontProperties(fname=font_path, size=10)
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

def scatter_plot(df):
    # 4개의 subplot을 위한 figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('주택 가격 에 영향을 주는 요소들의 산점도')
    # LSTAT - 저소득층 비율
    #     저소득층 인구 비율 (%) :  명확한 관계

    # TAX - 재산세율
    #     $10,000당 재산세율 : 재산세율 높으면 주택가격이 낮음

    # RAD - 고속도로 접근성
    #     방사형 고속도로 접근성 지수 :  높으면 주택가격 낮음

    # AGE - 노후 주택 비율
    #     1940년 이전에 건축된 주택의 비율 (%) : 높으면 주택가격이 주로 낮은 값을 가지는 경향
    
    # 예시: 4개의 feature를 선택하여 각각 그리기
    features = ['LSTAT', 'RM','AGE', 'TAX'] #'TAX', 'RAD', 
    target = 'MEDV'
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)
        axes[row, col].scatter(df[feature], df[target], alpha=0.7)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel(target)
        axes[row, col].set_title(f'{feature} vs {target}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def correlation_matrix(df):
    # 상관계수 계산
    correlation_matrix = df.corr(numeric_only=True)

    # MEDV와 다른 변수들의 상관계수만 추출 (자기 자신 제외)
    medv_corr = correlation_matrix["MEDV"].drop(labels="MEDV")

    # 그래프 크기 설정
    plt.figure(figsize=(6, 10))

    # 색상 설정 (음수는 파랑, 양수는 빨강)
    colors = medv_corr.apply(lambda x: 'red' if x > 0 else 'blue')

    # 수직 막대 그래프 그리기
    bars = plt.barh(medv_corr.index, medv_corr.values, color=colors)

    # 값 표시
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01 if width > 0 else width - 0.05,
                bar.get_y() + bar.get_height()/2,
                f"{width:.2f}",
                va='center',
                ha='left' if width > 0 else 'right')

    # 그래프 제목과 축
    plt.title("Correlation with MEDV")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.show()

def correlation_hitmap(df):
    import seaborn as sns
    # 상관계수 계산 (숫자형 변수들에 대해서만)
    correlation_matrix = df.corr(numeric_only=True)

    # MEDV와의 상관관계만 추출 (자기 자신 제외)
    # MEDV와의 상관관계만 추출하고 절대값 적용 (자기 자신 제외)
    medv_corr = correlation_matrix[["MEDV"]].drop(["MEDV", "CAT. MEDV"]).abs()
    
    # 절대값 기준으로 정렬 (내림차순)
    medv_corr = medv_corr.reindex(medv_corr.sort_values(by='MEDV',ascending=False).index)
    print(medv_corr)
    # 히트맵 시각화
    plt.figure(figsize=(6, 10))
    sns.heatmap(medv_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)

    # 제목 추가
    plt.title("Correlation with MEDV")
    plt.tight_layout()
    plt.show()

def polynomia(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures


    # X, y 정의
    X = df[["TAX"]].values
    y = df["MEDV"].values

    # 2차 다항 특징 생성
    poly = PolynomialFeatures(degree=2)  # ← 2차 비선형 회귀
    X_poly = poly.fit_transform(X)

    # 선형 회귀 모델로 적합
    model = LinearRegression()
    model.fit(X_poly, y)

    # 예측
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_pred = model.predict(X_range_poly)

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='gray', alpha=0.5, label='Data')
    plt.plot(X_range, y_pred, color='red', label='Polynomial Regression (deg=2)')
    plt.xlabel("TAX")
    plt.ylabel("MEDV")
    plt.title("TAX vs MEDV - Polynomial Regression (degree=2)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def pie_chart_cat_medv(df):
    # 'CAT. MEDV' 값의 비율 계산
    value_counts = df['CAT. MEDV'].value_counts()

    # 파이차트로 시각화
    plt.figure(figsize=(6,6))
    plt.pie(
        value_counts,
        labels=value_counts.index.map({0: '0 (Low)', 1: '1 (High)'}),
        autopct='%1.1f%%',
        startangle=90,
        colors=['skyblue', 'salmon']
    )
    plt.title("Proportion of CAT. MEDV (0 vs 1)")
    plt.axis('equal')  # 원형 유지
    plt.show()


def histogram_medv(df):
    # MEDV 히스토그램 그리기
    plt.figure(figsize=(8, 6))
    plt.hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of MEDV')
    plt.xlabel('MEDV (Median value of homes in $1000s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def relation_PTRATIO_LSTAT(df):
    # LSTAT 오름차순으로 정렬
    sorted_df = df.sort_values(by='LSTAT')

    # 선 그래프 그리기
    # LSTAT을 20개의 구간으로 나누기 (binning)
    df['LSTAT_bin'] = pd.cut(df['LSTAT'], bins=20)

    # 각 구간에서의 PTRATIO 평균 계산
    grouped = df.groupby('LSTAT_bin')['PTRATIO'].mean().reset_index()

    # 구간의 중앙값을 x축으로 사용
    grouped['LSTAT_bin_center'] = grouped['LSTAT_bin'].apply(lambda x: x.mid)

    print(grouped)
    # 선 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['LSTAT_bin_center'], grouped['PTRATIO'], marker='o', linestyle='-', color='green')
    plt.title('Average PTRATIO by LSTAT')
    plt.xlabel('LSTAT (Low-status population %)')
    plt.ylabel('Average PTRATIO')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("./BostonHousing.csv")

    # pie_chart_cat_medv(df)
    # histogram_medv(df)

    # print("데이터프레임의 첫 5행:")
    # print(df.head())
    correlation_hitmap(df)

    scatter_plot(df)
    relation_PTRATIO_LSTAT(df)


    

    


if __name__ == "__main__":
    main()