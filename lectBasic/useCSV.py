import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


plt.rcParams['font.family'] = 'AppleGothic'  # macOS

def load_csv(file_path):
    """CSV 파일을 읽어 DataFrame으로 반환하는 함수"""
    try:
        df = pd.read_csv(file_path)
        print(f"CSV 파일 '{file_path}'을(를) 성공적으로 읽었습니다.")
        return df
    except FileNotFoundError:
        print(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None
    except pd.errors.EmptyDataError:
        print("파일이 비어 있습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None 
    
def load_excel(file_path):
    """Excel 파일을 읽어 DataFrame으로 반환하는 함수"""
    try:
        df = pd.read_excel(file_path)
        print(f"Excel 파일 '{file_path}'을(를) 성공적으로 읽었습니다.")
        return df
    except FileNotFoundError:
        print(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None
    except pd.errors.EmptyDataError:
        print("파일이 비어 있습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None   
    


def excelTest():

    excel_file = 'data/Amtrak_data.xls'  # Excel 파일 경로
    df = load_excel(excel_file)  # Excel 파일 읽기
    if df is not None: 
        print(df.columns)  
    else:
        print("데이터를 불러오지 못했습니다.")
        return  

    print(f"\n 데이터 타입:")
    print(df.dtypes)

    # 날짜 컬럼을 datetime으로 변환
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'])
        print(" 날짜 컬럼 변환 완료")
    else:
        print(" 'Month' 컬럼을 찾을 수 없습니다.")

    # 승객 수 컬럼 확인
    ridership_col = None
    for col in df.columns:
        if 'ridership' in col.lower() or 'passenger' in col.lower():
            ridership_col = col
            break

    if ridership_col is None:
        # 숫자 컬럼 중 첫 번째를 승객 수로 사용
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ridership_col = numeric_cols[0]
            print(f" 승객 수 컬럼으로 '{ridership_col}' 사용")

    # 기본 통계
    print(f"\n 승객 수 기본 통계 (단위: 천명):")
    print(f"• 평균: {df[ridership_col].mean():.1f}")
    print(f"• 최대: {df[ridership_col].max():.1f}")
    print(f"• 최소: {df[ridership_col].min():.1f}")
    print(f"• 표준편차: {df[ridership_col].std():.1f}")


    def plot_ridership(df, ridership_col):
        # 월별 평균 계산
        df['Month_Num'] = df['Month'].dt.month
        monthly_avg = df.groupby('Month_Num')[ridership_col].mean()

        # 월별 계절성 그래프
        plt.figure(figsize=(12, 6))
        months = ['1월', '2월', '3월', '4월', '5월', '6월', 
                '7월', '8월', '9월', '10월', '11월', '12월']
        plt.plot(range(1, 13), monthly_avg.values, 'go-', linewidth=3, markersize=8)
        plt.title(' 월별 평균 승객 수 (계절성 분석)', fontsize=16, fontweight='bold')
        plt.xlabel('월', fontsize=12)
        plt.ylabel('평균 승객 수 (천명)', fontsize=12)
        plt.xticks(range(1, 13), months, rotation=45)
        plt.grid(True, alpha=0.3)

        # 최고/최저 월 표시
        max_month = monthly_avg.idxmax()
        min_month = monthly_avg.idxmin()
        plt.annotate(f'최고: {months[max_month-1]}\n{monthly_avg.max():.1f}천명', 
                    xy=(max_month, monthly_avg.max()), xytext=(max_month+1, monthly_avg.max()+50),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.annotate(f'최저: {months[min_month-1]}\n{monthly_avg.min():.1f}천명', 
                    xy=(min_month, monthly_avg.min()), xytext=(min_month+1, monthly_avg.min()-50),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()
        plt.show()

    def plot_ridership_trend(df, ridership_col):
        plt.figure(figsize=(14, 8))
        plt.plot(df['Month'], df[ridership_col], linewidth=2, color='steelblue', marker='o', markersize=3)
        plt.title('Amtrak ', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('year', fontsize=12)
        plt.ylabel('ridership', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 평균선 추가
        mean_ridership = df[ridership_col].mean()
        plt.axhline(y=mean_ridership, color='red', linestyle='--', alpha=0.7, 
                label=f'avg: {mean_ridership:.1f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_ridership_seasonality(df, ridership_col):
        # 연도별 데이터 집계
        df['Year'] = df['Month'].dt.year
        yearly_data = df.groupby('Year')[ridership_col].agg(['mean', 'sum', 'min', 'max']).round(1)

        print(" 연도별 승객 수 통계:")
        print(yearly_data.head(10))

        # 연도별 평균 승객 수 그래프
        plt.figure(figsize=(14, 6))
        plt.plot(yearly_data.index, yearly_data['mean'], 'o-', linewidth=3, markersize=8, color='darkblue')
        plt.title(' 연도별 평균 승객 수 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('평균 승객 수 (천명)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 추세선 추가
        z = np.polyfit(yearly_data.index, yearly_data['mean'], 1)
        p = np.poly1d(z)
        plt.plot(yearly_data.index, p(yearly_data.index), "r--", alpha=0.8, linewidth=2, 
                label=f'추세선 (기울기: {z[0]:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # plot_ridership_seasonality(df, ridership_col)
    plot_ridership_trend(df, ridership_col)
    # plot_ridership(df, ridership_col)

def csvTest():
    csv_file = 'data/daily_temperature_humidity.csv'  # CSV 파일 경로
    daily_data = load_csv(csv_file)  # CSV 파일 읽기
    if daily_data is not None: 
        daily_data['날짜'] = pd.to_datetime(daily_data['날짜'])
        print(daily_data.columns)  

    # 그리기
    plt.figure(figsize=(14, 8))

    # 첫 번째 축 (온도)
    ax1 = plt.gca()
    line1 = ax1.plot(daily_data['날짜'], daily_data['온도'], 
                    'ro-', linewidth=3, markersize=6, label='온도 (°C)')
    ax1.set_xlabel('날짜', fontsize=12)
    ax1.set_ylabel('온도 (°C)', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.tick_params(axis='x', rotation=45)

    # 두 번째 축 (습도)
    ax2 = ax1.twinx()
    line2 = ax2.plot(daily_data['날짜'], daily_data['습도'], 
                    'bs-', linewidth=3, markersize=6, label='습도 (%)')
    ax2.set_ylabel('습도 (%)', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')

    # 제목과 격자
    plt.title('온도와 습도 변화 (하나의 화면)', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    # 범례 합치기
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()

excelTest()