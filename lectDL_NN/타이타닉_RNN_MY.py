# 개선된 타이타닉 RNN 분류 모델
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (한국어 환경에서 사용 시)
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 전처리 개선
def preprocess_data():
    """데이터 전처리 함수"""
    # 데이터 로드
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # 필요한 컬럼 선택
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target_col = 'Survived'
    df = df[cols + [target_col]].copy()
    
    # 결측치 처리 개선
    # Age: 중앙값으로 대체 (평균값보다 이상치에 덜 민감)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fare: 중앙값으로 대체
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Embarked: 최빈값으로 대체
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 파생 변수 생성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # 가족 규모
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)  # 혼자 여행 여부
    
    # 나이 그룹화
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=[0, 1, 2, 3, 4])
    
    # 요금 그룹화
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3])
    
    # 범주형 변수 인코딩
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    # 최종 feature 선택 (파생 변수 포함)
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Embarked', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)
    
    # 수치형 변수 정규화
    scaler = StandardScaler()
    # Age, Fare, FamilySize 컬럼 정규화
    numeric_indices = [2, 5, 7]  # Age, Fare, FamilySize 인덱스
    X[:, numeric_indices] = scaler.fit_transform(X[:, numeric_indices])
    
    return X, y, scaler, le_sex, le_embarked

# 2. 개선된 Dataset 클래스
class TitanicSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. 개선된 RNN 모델 (LSTM + Dropout + Batch Normalization)
class ImprovedTitanicRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, num_classes=2, dropout=0.3):
        super(ImprovedTitanicRNN, self).__init__()
        
        # LSTM 사용 (RNN보다 성능이 좋음)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # 분류층 개선
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 마지막 시퀀스의 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # Batch Normalization 적용
        normalized = self.batch_norm(last_output)
        
        # 분류
        output = self.classifier(normalized)
        
        return output

# 4. 학습 함수
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """모델 학습 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 옵티마이저와 손실함수 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # 학습 기록
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (그래디언트 폭발 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 통계 업데이트
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # 검증 모드
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # 평균 계산
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # 기록 저장
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # 진행상황 출력
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.3f}")
    
    # 최고 성능 모델 로드
    model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

# 5. 평가 함수
def evaluate_model(model, test_loader):
    """모델 평가 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

# 6. 시각화 함수
def plot_training_history(history):
    """학습 기록 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    ax1.plot(history['train_losses'], label='Train Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 정확도 그래프
    ax2.plot(history['train_accs'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_accs'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix - Improved RNN Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 7. 메인 실행 코드
def main():
    print("타이타닉 생존 예측 - 개선된 RNN 모델")
    print("=" * 50)
    
    # 데이터 전처리
    print("1. 데이터 전처리 중...")
    X, y, scaler, le_sex, le_embarked = preprocess_data()
    
    # RNN을 위한 시퀀스 형태로 변환
    X_sequence = X.reshape(X.shape[0], X.shape[1], 1)
    print(f"데이터 형태: {X_sequence.shape}")
    
    # 데이터 분할 (train/val/test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_sequence, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"훈련 데이터: {X_train.shape[0]}개")
    print(f"검증 데이터: {X_val.shape[0]}개")
    print(f"테스트 데이터: {X_test.shape[0]}개")
    
    # 데이터 로더 생성
    train_dataset = TitanicSequenceDataset(X_train, y_train)
    val_dataset = TitanicSequenceDataset(X_val, y_val)
    test_dataset = TitanicSequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 모델 생성
    print("\n2. 모델 생성 중...")
    model = ImprovedTitanicRNN(
        input_size=1, 
        hidden_size=64, 
        num_layers=2, 
        num_classes=2, 
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    
    # 모델 학습
    print("\n3. 모델 학습 중...")
    history = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    
    print(f"\n최고 검증 정확도: {history['best_val_acc']:.4f}")
    
    # 학습 기록 시각화
    print("\n4. 학습 기록 시각화...")
    plot_training_history(history)
    
    # 테스트 데이터 평가
    print("\n5. 테스트 데이터 평가...")
    predictions, true_labels, probabilities = evaluate_model(model, test_loader)
    
    test_accuracy = (predictions == true_labels).mean()
    print(f"테스트 정확도: {test_accuracy:.4f}")
    
    # 상세 성능 리포트
    print("\n=== 상세 성능 리포트 ===")
    print(classification_report(true_labels, predictions, 
                              target_names=['Not Survived', 'Survived'], digits=4))
    
    # 혼동 행렬 시각화
    plot_confusion_matrix(true_labels, predictions)
    
    return model, history, (predictions, true_labels, probabilities)

# 실행
if __name__ == "__main__":
    model, history, results = main()