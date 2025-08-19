import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import seaborn as sns


def set_font():
    font_path='/System/Library/Fonts/AppleGothic.ttf'
    # fontprop = fm.FontProperties(fname=font_path, size=10)
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS
    plt.rcParams['axes.unicode_minus'] = False
    
data_path = './data/국민건강보험공단_건강검진정보_2023.CSV'
output_csv = './data/sampled_10000_data.csv'
cleaned_csv ='./data/cleaned_data.csv'

def split_csv():
    # 원본 CSV 파일 경로
    input_csv = data_path

    # 출력할 CSV 파일 경로
    output_csv = './data/sampled_10000_data.csv'

    # 상위 100개만 읽어서 저장
    df = pd.read_csv(input_csv, encoding="euc-kr",nrows=10000)
    df.to_csv(output_csv, index=False)

    print(f"{len(df)}개의 데이터를 '{output_csv}'에 저장했습니다.")

    df = pd.read_csv(data_path, encoding="euc-kr")
    df['BMI'] = df['체중(5kg단위)'] / ((df['신장(5cm단위)'] / 100) ** 2)

    removes = ['구강검진수검여부','치아우식증유무','결손치 유무','치아마모증유무','제3대구치(사랑니) 이상','치석' ]
    df = df.drop(columns=removes)

    features = ['성별코드', '연령대코드(5세단위)', 'BMI', '허리둘레', '혈청크레아티닌','혈청지오티(AST)','혈청지피티(ALT)', '흡연상태']
    features = ['성별코드', 'BMI', '허리둘레', '혈청지오티(AST)','혈청지피티(ALT)', '흡연상태']
    
    features.append('감마지티피')
    df = df[features].dropna()

    df.to_csv(cleaned_csv, index=False)

    
def preprocess_standardize(df):
    # scaler = StandardScaler()
    # df['감마지티피'] = scaler.fit_transform(df[['감마지티피']])

    # 1. 감마지티피는 제외한 X만 추출
    target_col = '감마지티피'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. 원-핫 인코딩 (범주형 변수가 있다면)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 3. StandardScaler 적용
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # 4. DataFrame으로 다시 변환 (컬럼명 유지)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X.index)

    # 5. 감마지티피 추가
    X_scaled_df[target_col] = y

    # 5. 결과 확인 (선택)
    print(X_scaled_df.head())
    return X_scaled_df

def preprocess_nomalization(df):
    scaler = MinMaxScaler()
    df['감마지티피'] = scaler.fit_transform(df[['감마지티피']])
    return df


def preprocess_outlier(df):
    method = 1
    if method == 0:
        # 1. IQR 기반 이상치 기준 계산
        Q1 = df['감마지티피'].quantile(0.10)
        Q3 = df['감마지티피'].quantile(0.80)
        # Q3 = df['감마지티피'].quantile(0.85)
        IQR = Q3 - Q1

        # 2. 이상치 범위 정의
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"q1: {Q1}, q3: {Q3}, iqr: {IQR}, lower: {lower_bound}, upper: {upper_bound}")
        # 3. 이상치가 아닌 데이터만 필터링
        df_cleaned = df[(df['감마지티피'] >= lower_bound) & (df['감마지티피'] <= upper_bound)]

        # (선택) 제거 전/후 비교
        print(f"원본 행 개수: {len(df)}")
        print(f"이상치 제거 후 행 개수: {len(df_cleaned)}")

        print(df_cleaned.describe())
    elif method == 1:
        df_filtered = df[df['감마지티피'] <= 900]  # 예: 도메인에서 300 이상은 비정상으로 간주
        return df_filtered
    else:
        median = np.median(df['감마지티피'])
        mad = np.median(np.abs(df['감마지티피'] - median))
        modified_z = 0.6745 * (df['감마지티피'] - median) / mad
        df_modz_filtered = df[abs(modified_z) < 3.5]
        print('##################\n')
        print(df_modz_filtered.describe())
        return df_modz_filtered
    return df_cleaned


def load_data():
    # 1. 데이터 로드
    # df = pd.read_csv(data_path, encoding="euc-kr")
    df = pd.read_csv(cleaned_csv, encoding="utf-8")
    # df['BMI'] = df['체중(5kg단위)'] / ((df['신장(5cm단위)'] / 100) ** 2)
    removes = ['구강검진수검여부','치아우식증유무','결손치 유무','치아마모증유무','제3대구치(사랑니) 이상','치석' ]
    # df = df.drop(columns=removes)

    features = ['성별코드', '연령대코드(5세단위)', 'BMI', '허리둘레', '혈청크레아티닌','혈청지오티(AST)','혈청지피티(ALT)', '흡연상태']
    features = ['성별코드', 'BMI', '허리둘레', '혈청지오티(AST)','혈청지피티(ALT)', '흡연상태']
    
    features.append('감마지티피')
    df_cleaned = df[features].dropna()


    print(df_cleaned[['감마지티피']].describe())
    # df_cleaned = preprocess_outlier(df_cleaned)

    df_cleaned = preprocess_standardize(df)
    print(df_cleaned[['감마지티피']].describe())

    # df_cleaned = preprocess_nomalization(df_cleaned)
    # print(df_cleaned[['감마지티피']].describe())

    df_cleaned = preprocess_outlier(df_cleaned)
    print(df_cleaned[['감마지티피']].describe())
    return df_cleaned

def corr_show(df):
    # 숫자형 데이터만 추출
    numeric_df = df.select_dtypes(include='number')

    # 상관계수 행렬 계산
    corr_matrix = numeric_df.corr()

    # 감마지티피와의 상관계수만 추출 (내림차순 정렬)
    gamma_corr = corr_matrix['감마지티피'].drop('감마지티피').abs().sort_values(ascending=False)

    # 히트맵 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(gamma_corr.to_frame(), annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("감마지티피와 다른 변수들과의 상관관계")
    plt.tight_layout()
    plt.show()

def make_models(df):

    target_col = '감마지티피'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # 3. 모델 정의 및 하이퍼파라미터 탐색

    # Linear Regression
    models = [("Linear Regression", LinearRegression())]

    # Polynomial Regression (degree=2, 3)
    for degree in [2, 3]:
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('reg', LinearRegression())
        ])
        models.append((f"Polynomial Regression (deg={degree})", poly_model))

    # Ridge Regression
    ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5)
    ridge_grid.fit(X_train, y_train)
    models.append((f"Ridge Regression (α={ridge_grid.best_params_['alpha']})", ridge_grid.best_estimator_))

    # Lasso Regression
    lasso_params = {'alpha': [0.01, 0.1, 1, 10]}
    lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5)
    lasso_grid.fit(X_train, y_train)
    models.append((f"Lasso Regression (α={lasso_grid.best_params_['alpha']})", lasso_grid.best_estimator_))

    # KNeighbors Regressor
    knn_params = {'n_neighbors': range(1, 11)}
    knn_grid = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5)
    knn_grid.fit(X_train, y_train)
    models.append((f"KNN Regressor (k={knn_grid.best_params_['n_neighbors']})", knn_grid.best_estimator_))

    # Decision Tree Regressor
    tree_params = {'max_depth': [3, 5, 10, 15, None]}
    tree_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), tree_params, cv=5)
    tree_grid.fit(X_train, y_train)
    models.append((f"Decision Tree (depth={tree_grid.best_params_['max_depth']})", tree_grid.best_estimator_))

    # 4. 평가 함수 정의
    def evaluate_model(name, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred)
        }

    # 5. 모든 모델 평가
    results = [evaluate_model(name, model, X_train, y_train, X_test, y_test)
            for name, model in models]
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)

    # 6. 결과 출력
    print(results_df)

    # 7. 성능 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='R2', y='Model', palette='viridis')
    plt.title("감마지티피 예측 - 회귀 모델별 R² 성능 비교")
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


    # # 8. 예측 vs 실제값 시각화 (상위 3개 모델 기준)
    top_n = 3
    plt.figure(figsize=(15, 4))

    top_models = results_df.head(top_n)['Model'].values  # 상위 모델 이름 리스트만 추출

    for i, name in enumerate(top_models):
        # 모델 이름으로 models 리스트에서 해당 모델 객체 찾기
        model_obj = next(m for m in models if m[0] == name)[1]
        y_pred = model_obj.predict(X_test)

        plt.subplot(1, top_n, i + 1)
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y = x 선
        plt.xlabel("실제값 (감마지티피)")
        plt.ylabel("예측값")
        plt.title(f"{name}")
        plt.grid(True)

    plt.suptitle("모델별 예측값 vs 실제값")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_font()
    # split_csv()
    df = load_data()
    # corr_show(df)
    make_models(df)

