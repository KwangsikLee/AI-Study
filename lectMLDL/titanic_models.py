import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def set_font():
    font_path='/System/Library/Fonts/AppleGothic.ttf'
    # fontprop = fm.FontProperties(fname=font_path, size=10)
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS
    plt.rcParams['axes.unicode_minus'] = False

data_path = './data/titanic_kor.xlsx'

def mainLogic():

    # 1. 데이터 로딩
    df = pd.read_excel(data_path)

    # 2. 전처리
    df = df.drop(columns=['이름', '티켓', '승객번호', '객실', '생존여부'], errors='ignore')
    df['성별'] = df['성별'].map({'male': 0, 'female': 1, '남자': 0, '여자': 1})
    df['탑승항'] = df['탑승지명'].fillna('S')
    df = pd.get_dummies(df, columns=['탑승항'])
    for col in ['나이', '요금']:
        df[col] = df[col].fillna(df[col].median())

    # 3. 학습 데이터 준비
    target_col = '생존'
    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. 모델 및 하이퍼파라미터
    model_grids = {
        'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
        # 'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
        # 'SVM': (SVC(probability=True), {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}),
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [None, 5, 10]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
        # 'Naive Bayes': (GaussianNB(), {})  # 튜닝 없음
    }

    # 5. 평가 및 결과 수집
    results = {}
    for name, (model, params) in model_grids.items():
        if params:
            grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            # 'ROC AUC': roc_auc_score(y_test, y_prob)
        }

    # 6. 결과 정리 및 시각화
    results_df = pd.DataFrame(results).T.round(3)
    print(results_df.sort_values(by='F1 Score', ascending=False))

    # 시각화
    results_df[['Accuracy', 'F1 Score', 'Precision']].plot(kind='bar', figsize=(14, 6), grid=True)
    plt.title("Titanic 생존율 예측 모델 성능 비교")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_font()
    mainLogic()


