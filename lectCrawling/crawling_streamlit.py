# 필요한 라이브러리 임포트
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

import streamlit as st
import plotly.express as px


def crawl_naver_news():
    """네이버 뉴스 메인 페이지에서 헤드라인을 크롤링하는 함수"""

    print(" 네이버 뉴스 크롤링을 시작합니다...")

    # 웹 페이지 URL
    url = "https://news.naver.com"

    # 헤더 설정 (코랩 환경에 맞게 조정)
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # 웹 페이지 요청
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # 뉴스 헤드라인 추출
        headlines = []

        # 여러 가지 방법으로 뉴스 헤드라인 찾기
        selectors = [
            'a.cjs_news_link',
            'a[href*="/article/"]',
            '.hdline_article_tit',
            '.cluster_text_headline'
        ]

        for selector in selectors:
            items = soup.select(selector)
            if items:
                for item in items[:10]:
                    title = item.get_text().strip()
                    if title and len(title) > 10:
                        headlines.append(title)
                break

        # 결과 출력
        print(f"\n 총 {len(headlines)}개의 뉴스 헤드라인을 찾았습니다!")
        print("=" * 60)

        for i, headline in enumerate(headlines[:10], 1):
            print(f"{i:2d}. {headline}")

        print("=" * 60)

        return headlines

    except requests.RequestException as e:
        print(f" 네트워크 오류: {e}")
        return []
    except Exception as e:
        print(f" 크롤링 오류: {e}")
        return []
    

def create_news_dataframe(headlines):
    """뉴스 헤드라인을 데이터프레임으로 변환"""

    if not headlines:
        print(" 분석할 헤드라인이 없습니다.")
        return None

    # 데이터프레임 생성
    df = pd.DataFrame({
        '순번': range(1, len(headlines) + 1),
        '헤드라인': headlines,
        '글자수': [len(headline) for headline in headlines],
        '크롤링시간': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * len(headlines)
    })

    print("\n 뉴스 데이터 분석:")
    print(f"총 뉴스 개수: {len(df)}")
    print(f"평균 글자수: {df['글자수'].mean():.1f}")
    print(f"최대 글자수: {df['글자수'].max()}")
    print(f"최소 글자수: {df['글자수'].min()}")

    return df

def visualize_news_data(df):
    """뉴스 데이터 시각화"""

    if df is None or df.empty:
        print(" 시각화할 데이터가 없습니다.")
        return

    # 그래프 스타일 설정
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 헤드라인 글자수 분포
    ax1.hist(df['글자수'], bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('News Headline Length Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Characters')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # 2. 헤드라인별 글자수
    ax2.bar(df['순번'], df['글자수'], color='lightcoral', alpha=0.7)
    ax2.set_title('Character Count by Headline', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Headline Number')
    ax2.set_ylabel('Number of Characters')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_streamlt(df):    

    st.write("\n 뉴스 데이터 분석:")
    st.write(f"총 뉴스 개수: {len(df)}")
    st.write(f"평균 글자수: {df['글자수'].mean():.1f}")
    st.write(f"최대 글자수: {df['글자수'].max()}")
    st.write(f"최소 글자수: {df['글자수'].min()}")

    # 그래프 스타일 설정
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 헤드라인 글자수 분포
    ax1.hist(df['글자수'], bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('News Headline Length Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Characters')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # 2. 헤드라인별 글자수
    ax2.bar(df['순번'], df['글자수'], color='lightcoral', alpha=0.7)
    ax2.set_title('Character Count by Headline', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Headline Number')
    ax2.set_ylabel('Number of Characters')
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

        # 1. 헤드라인 글자수 분포
    fig2 = px.line(df, x="순번", y="글자수", title="헤드라인 글자수", markers=True)
    st.plotly_chart(fig2, use_container_width=True)



# 메인 실행 함수
def main():
    """메인 실행 함수"""

    print(" 구글 코랩 웹 크롤링 실습을 시작합니다!")
    print("=" * 70)

    # 1. 기초 개념 데모
    #demo_web_scraping_basics()

    print("\n" + "="*70)

    # 2. 실제 웹사이트 크롤링
    headlines = crawl_naver_news()

    # 3. 간단한 테스트 사이트 크롤링
    #crawl_simple_news_site()

    # 4. 데이터 분석
    if headlines:
        df = create_news_dataframe(headlines)

        if df is not None:
            print("\n 데이터프레임 미리보기:")
            print(df.head())

            # 5. 시각화
            #visualize_news_data(df)
            show_streamlt(df)

            # 6. CSV 파일로 저장 (코랩에서 다운로드 가능)
            df.to_csv('news_headlines.csv', index=False, encoding='utf-8')
            print("\n 'news_headlines.csv' 파일로 저장되었습니다!")
            print("   (코랩 왼쪽 파일 탭에서 다운로드 가능)")

    print("\n 크롤링 실습이 완료되었습니다!")



# 코랩에서 바로 실행
if __name__ == "__main__":    
    main()

# To run this script in Streamlit, use the command:
#streamlit run ./lectCrawling/crawling_streamlit.py 
# 참고 Streamlit 문서: https://docs.streamlit.io/library/get-started
# https://wikidocs.net/book/17846