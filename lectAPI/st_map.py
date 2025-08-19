import pandas as pd
import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
import matplotlib.font_manager as fm
import webbrowser
import streamlit as st
from streamlit_folium import folium_static


plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
df = pd.read_excel('./lectAPI/학교주소좌표.xlsx')
df.columns = ['학교명', '주소', '경도', '위도']

# 2. 시군구 추출 함수 정의
def extract_city_district(address):
    parts = address.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else address

df['시군구'] = df['주소'].apply(extract_city_district)

# 3. 시군구별 학교 수 집계
district_counts = df['시군구'].value_counts().sort_values(ascending=False)

def create_bar_chart(df):
    # ───────────────────────────────
    #  시군구별 학교 수 막대 그래프
    fig = plt.figure(figsize=(14, 8))
    district_counts[:20].plot(kind='bar')
    plt.title('시군구별  대학 수 (상위 20개)')
    plt.xlabel('시군구')
    plt.ylabel('대학 수')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.show()
    st.pyplot(fig)  # Streamlit에서 그래프 표시



def create_pie_chart(df):
    # ───────────────────────────────
    # 시군구별 파이 차트 생성
    fig = plt.figure(figsize=(10, 10))
    df['시군구'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('시군구별 대학 분포 비율')
    plt.ylabel('')
    plt.tight_layout()
    # plt.show()  
    st.pyplot(fig)  # Streamlit에서 파이 차트 표시  


def create_heatmap(df):
    # ───────────────────────────────
    #  지도 히트맵 시각화
    map_center = [df['위도'].mean(), df['경도'].mean()]
    heatmap = folium.Map(location=map_center, zoom_start=7)

    heat_data = df[['위도', '경도']].values.tolist()
    HeatMap(heat_data).add_to(heatmap)

    # 지도 결과 반환
    heatmap.save('heatmap.html')
    folium_static(heatmap)  # Streamlit에서 히트맵 표시


def main():
    # ───────────────────────────────
    #  시각화 함수 호출
    create_bar_chart(df)
    create_pie_chart(df)
    heatmap = create_heatmap(df)


if __name__ == "__main__":
    main()    