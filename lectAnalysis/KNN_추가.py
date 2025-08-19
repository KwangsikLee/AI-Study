# 1) 라이브러리 불러오기
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2) 데이터 로드 및 전처리
mpg = sns.load_dataset('mpg').dropna()
df = mpg[['horsepower', 'weight', 'displacement', 'mpg']]

# 3) 데이터 확인
print("■ First 5 rows:")
display(df.head())
print("\n■ Summary statistics:")
display(df.describe().T)

# 4) 상관관계 분석
corr = df.corr()
print("\n■ Feature–Target Correlations:")
corr_with_target = corr['mpg'].drop('mpg').sort_values(ascending=False)
display(corr_with_target.to_frame('corr_with_target'))

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 5) 설명 변수/타깃 분리
X = df[['horsepower', 'weight', 'displacement']]
y = df['mpg']

# 6) 학습/테스트 분할 (30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 7) OLS 회귀 (Statsmodels)
X_const = sm.add_constant(X)            # 전체 데이터 상수항 포함
ols_model = sm.OLS(y, X_const).fit()
print("\n■ OLS Regression Results:")
print(ols_model.summary())

# 8) KNN 회귀 모델 학습
knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=2)
knn.fit(X_train, y_train)

# 9) 예측
y_pred_ols = ols_model.predict(sm.add_constant(X_test))
y_pred_knn = knn.predict(X_test)

# 10) 평가 지표 계산
def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{name}  RMSE: {rmse:.3f}, R²: {r2:.3f}")

print("\n■ Model Performance:")
print_metrics("OLS", y_test,    y_pred_ols)
print_metrics("KNN", y_test,    y_pred_knn)

# 11) Residuals vs Predicted Plot
plt.figure(figsize=(10,4))
for i,(name,y_p) in enumerate([('OLS', y_pred_ols), ('KNN', y_pred_knn)]):
    plt.subplot(1,2,i+1)
    plt.scatter(y_p, y_test - y_p, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{name} Residuals')
    plt.xlabel('Predicted MPG')
    plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

# 12) Actual vs Predicted Plot
plt.figure(figsize=(10,4))
for i,(name,y_p) in enumerate([('OLS', y_pred_ols), ('KNN', y_pred_knn)]):
    plt.subplot(1,2,i+1)
    plt.scatter(y_test, y_p, alpha=0.5)
    m = max(y_test.max(), y_p.max())
    plt.plot([0, m], [0, m], 'r--')
    plt.title(f'{name} Actual vs Predicted')
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
plt.tight_layout()
plt.show()

# 13) Feature vs Predicted for Each Model (상관계수 높은 순)
features_sorted = corr_with_target.index.tolist()
plt.figure(figsize=(12, 4*len(features_sorted)))
for idx, feat in enumerate(features_sorted):
    plt.subplot(len(features_sorted), 1, idx+1)
    plt.scatter(X_test[feat], y_test,      label='Actual', alpha=0.5)
    plt.scatter(X_test[feat], y_pred_ols,  label='OLS Pred', alpha=0.5)
    plt.scatter(X_test[feat], y_pred_knn,  label='KNN Pred', alpha=0.5)
    plt.title(f'{feat} vs MPG (corr={corr.loc[feat,"mpg"]:.2f})')
    plt.xlabel(feat)
    plt.ylabel('MPG')
    if idx == 0:
        plt.legend()
plt.tight_layout()
plt.show()