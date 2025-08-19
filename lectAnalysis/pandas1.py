import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def pandas1():
    """
    pandas1.py에서 pandas를 사용하여 데이터프레임을 생성하고 출력하는 함수
    :return: None
    """

    city = { '서울': 100, '부산': 200, '대구': 150, '인천': 80, '광주': 90 }
    df = pd.Series(city, index=['서울', '부산'])

    # 데이터프레임 생성
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }
    
    df = pd.DataFrame(data)
    # print(df)
    # # 데이터프레임 출력
    # print(df.loc[0])
    # print(df.loc[[0]])


    data = {
        '도시': ['서울', '부산', '울산'],
        '인구': [25, 30, 35]
    }
    
    df = pd.DataFrame(data, index=['a', 'b', 'c'])
    print(df)


def pandas2():
    """
    pandas2.py에서 pandas를 사용하여 데이터프레임을 생성하고 출력하는 함수
    :return: None
    """

    data = [['김민재', 27, 75, 5428000],
            ['이강인', 22, 57, 3428000],
            ['박찬호', 50, 91, 8428000],
            ['차범근', 70, 80, 4428000],
            ['추신수', 43, 100, 4528000],
            ['손흥민', 31, 72, 7028000],
            ['황희찬', 28, 69, 2528000]]

    df = pd.DataFrame(data, columns=['성명', '나이', '몸무게', '급여'])



    # 데이터프레임 출력
    # print(df[['성명', '나이', '몸무게']])  # 특정 열만 데이터프레임 형태로 출력
    # print(df.loc[0])  # 첫 번째 행 출력
    # print(df.loc[[0]])  # 첫 번째 행을 데이터프레임 형태로 출력
    print(df.loc[0:2])
    # print(df.loc[[0]])


def pandas3():
    """
    pandas3.py에서 pandas를 사용하여 데이터프레임을 생성하고 출력하는 함수
    :return: None
    """

    data = {
        '이름': ['김민재', '이강인', '박찬호'],
        '나이': [27, 22, 50],
        '몸무게': [75, 57, 91],
        '급여': [5428000, 3428000, 8428000]
    }

    df = pd.DataFrame(data)
    
    # print(df)
    # 데이터프레임 출력
    # print(df.loc[0])
    # print(df.loc[[0]])
    
    print((df['급여'] >= 5000000) & (df['나이'] >= 30 )) # 특정 조건을 만족하는 행을 선택
    print(df.loc[(df['급여'] >= 5000000) & (df['나이'] >= 30 )]) # 특정 조건을 만족하는 행을 선택
    print(df.loc[[True, False, True]])  

    for i in range(len(df)):
        print(f"이름: {df.loc[i, '이름']}, 나이: {df.loc[i, '나이']}, 몸무게: {df.loc[i, '몸무게']}, 급여: {df.loc[i, '급여']}")    


def pandas_csv():
    """
    pandas_csv.py에서 pandas를 사용하여 CSV 파일을 읽고 출력하는 함수
    :return: None
    """
    
    # CSV 파일 읽기
    df = pd.read_csv('pandas_groupby.csv', encoding='euc-kr')
    
    # print(df['지역'].unique())
    max_sales = df.groupby('지역')['판매액'].transform('max')
    # print(max_sales)
    # print(df[df['판매액'] == max_sales])
    # print(df.groupby('지역')['판매액'].max())  # 지역별로 그룹화하여 합계 계산

    # print(df.groupby(['지역', '제품'])['판매액'].sum())  # 지역과 제품별로 그룹화하여 판매액의 합계를 계산
    # print(df.groupby(['판매원', '제품'])[['판매액', '판매개수']].sum())  # 지역과 제품별로 그룹화하여 판매액의 합계를 계산

    # print(df.groupby(['지역', '제품']).agg({'max'}))  #

    print(df.columns[3])
    # 데이터프레임 
    # print(df)
    

def pandas_titanic():
    """
    pandas_titanic.py에서 pandas를 사용하여 타이타닉 데이터셋을 읽고 데이타 분석
    :return: None
    """
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv('titanic_kor.csv', encoding='utf-8')
    except UnicodeDecodeError:
        # euc-kr 인코딩으로 다시 시도
        print("utf-8 인코딩으로 읽기 실패, euc-kr 인코딩으로 다시 시도합니다.")
        # euc-kr 인코딩으로 CSV 파일 읽기
        df = pd.read_csv('titanic_kor.csv', encoding='euc-kr')
    
    # print(df.head())  # 데이터프레임의 처음 5행 출력
    # print(df.info())  # 데이터프레임의 정보 출력
    # print(df.describe())  # 데이터프레임의 통계 요약 출력

    # print("컬럼별 결측치 개수:")
    # print(df.isnull().sum())  # 각 컬럼의 결측치 개수  

    print(" 성인여부 변경 전 유니크 값:")
    print(df['성인여부'].unique())  # 성인여부 컬럼의 유니크 값 출력
    df['성인여부'] = df['성인여부'].replace({'man': 'adult', 'woman': 'adult'})
    print(df['성인여부'].unique())  # 성인여부 컬럼의 유니크 값 출력

    print(" 객실등급 변경 전 유니크 값:")
    print(df['객실등급'].unique())  # 객실등급 컬럼의 유니크 값 출력
    df['객실등급'] = df['객실등급'].str.lower()  # 객실등급을 소문자로 변환
    print(df['객실등급'].unique())  # 객실등급 컬럼의 유니크 값 출력

    print(" 객실 요금>= 100 $인 데이터")
    high_fare = df[df['요금'] >= 300]
    print(high_fare.loc[:, ['요금','성별', '나이', '객실등급', '생존여부']])  # 객실 요금이 100 이상인 데이터 출력

    print(" 생존자 데이타")
    survivors = df[df['생존여부'] == 'yes']
    print(survivors)  # 생존자 데이터 출력

    print("객실 등급별 승객수   ")
    print(df['객실등급'].value_counts())  # 객실 등급별 승객 수 출력. Return a Series containing counts of unique values.

    print("객실 등급별 평균나이")
    print(df.groupby('객실등급')['나이'].mean())  # 객실 등급별 평균 나이 출력

    print("결측치 확인")
    print(df.isnull().sum())  # 각 컬럼의 결측치 개수 출력
    print(len(df))  # 결측치 확인 결과 출력
    miss_df = df['탑승지코드'].isnull()  # 성인여부 컬럼의 결측치 확인
    list_miss_df = df.index[miss_df].tolist()  # 결측치 확인 결과를 리스트로 변환
    df.dropna(subset=['탑승지코드'], inplace=True)  # 결측치가 있는 행 제거
    print(len(df))  # 결측치 확인 결과 출력
    
    sorted_df = df.sort_values(by=['생존', '요금', '성인여부'], ascending=[False, False, False])  # 나이 컬럼을 기준으로 내림차순 정렬
    # print(sorted_df.head(10))  # 정렬된 데이터프레임의 처음 10행 출력

    ##### 분석 
    print("성별과 성인여부에 따른 생존 여부")
    print(df.groupby(['성별', '성인여부'])['생존'].mean())  # 성별과 성인여부에 따른 생존율 출력
    # print(df.groupby(['생존여부','성인여부'])['생존'].to_list())  # 

    f_surv = df[df['성별'] == 'female']['생존'].mean()
    m_surv = df[df['성별'] == 'male']['생존'].mean()  #
    print(f"여성 생존율: {f_surv:.2f}, 남성 생존율: {m_surv:.2f}")  # 여성과 남성의 생존율 출력

    a_surv = df[(df['성인여부'] == 'adult') & (df['요금'] > 100)]['생존'].mean()
    c_surv = df[(df['성인여부'] == 'child') & (df['요금'] > 100)]['생존'].mean()  #
    print(f"성인 생존율: {a_surv:.2f}, 아이 생존율: {c_surv:.2f}")  # 성인과 아이의 생존율 출력

    #### 나누어진 데이타로 분석
    df_model = df[['성별', '나이', '요금', '객실등급', '생존']].dropna()  # 결측치가 있는 행 제거
    print("모델링을 위한 데이터프레임")
    print(df_model.head())  # 모델링을 위한 데이터프레임 출력   


    # 레이블 인코딩 : 문자열들을 코드화    
    sk_code1 = LabelEncoder()
    df_model['성별_encode'] = sk_code1.fit_transform(df_model['성별'])
    df_model['객실등급_encode'] = sk_code1.fit_transform(df_model['객실등급'])

    # 나이 컬럼을 기준으로 연령대 구분
    df_model['AgeBin'] = pd.cut(df_model['나이'], bins=[0, 12, 18, 35, 60, 120], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    print("연령대 구분된 데이터프레임")
    print(df_model.groupby('AgeBin')['생존'].mean())  # 연령대 구분된 데이터프레임 출력
    df_model['AgeBin_encode'] = sk_code1.fit_transform(df_model['AgeBin'])

    # print("AgeBin과 AgeBin_encode 컬럼의 유니크 값")
    # print(df_model[['AgeBin', 'AgeBin_encode']].drop_duplicates())  #

    ### Logistic Regression 모델링
    X = df_model[['성별_encode', '나이', '요금', '객실등급_encode']]
    y = df_model['생존']    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 데이터 분할
    model = LogisticRegression(max_iter=1000)  # 로지스틱 회귀 모델 생성
    model.fit(X_train, y_train)  # 모델 학습
    y_pred = model.predict(X_test)  # 예측      
    accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
    print(f"모델 정확도: {accuracy:.2f}")  # 모델 정확도 출력
    
    print("=== Classification Report ===")
    print("분류 보고서:")
    print(classification_report(y_test, y_pred))  # 분류 보고서 출력    



    # print("레이블 인코딩된 데이터프레임")
    # print(df_model.tail(30))  # 레이블 인코딩된 데이터프레임 출력 

if __name__ == "__main__":
    pandas_titanic()
    # pandas1.py를 실행하면 데이터프레임이 출력됩니다.
    # 이 코드는 pandas 라이브러리를 사용하여 간단한 데이터프레임을 생성하고 출력하는 예제입니다.    