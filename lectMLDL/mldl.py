
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score

def knn_visualization():
    # 1) 샘플 데이터 생성 (1차원 비선형 함수 + 노이즈)
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel() + 0.2 * np.random.randn(40)

    # 2) 테스트용 x축 데이터 생성
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)

    # 3) 두 모델: uniform (기본), distance (가중치 기반)
    models = {
        "K=3, uniform": KNeighborsRegressor(n_neighbors=3, weights='uniform'),
        "K=3, distance": KNeighborsRegressor(n_neighbors=3, weights='distance'),
        "K=10, uniform": KNeighborsRegressor(n_neighbors=10, weights='uniform'),
        "K=10, distance": KNeighborsRegressor(n_neighbors=10, weights='distance')
    }

    # 4) 시각화
    plt.figure(figsize=(12, 8))

    for i, (label, model) in enumerate(models.items(), 1):
        model.fit(X, y)
        y_pred = model.predict(X_test)

        plt.subplot(2, 2, i)
        plt.scatter(X, y, color='gray', label='Train data')
        plt.plot(X_test, y_pred, color='blue', label='KNN prediction')
        plt.title(label)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

    plt.tight_layout()
    plt.show()
        

def knn_cross():
    from sklearn.datasets import load_iris

    # 데이터 로드
    X, y = load_iris(return_X_y=True)
    k_range = range(1, 21)
    cv_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        cv_scores.append(scores.mean())

    # 결과 시각화
    plt.plot(k_range, cv_scores, marker='o')
    plt.xlabel('K value')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('KNN Cross-Validation Accuracy vs K')
    plt.grid(True)
    plt.show()

    # 최적 K 출력
    best_k = k_range[cv_scores.index(max(cv_scores))]
    print("최적의 K:", best_k)        


    


if __name__ == "__main__":
    knn_cross()