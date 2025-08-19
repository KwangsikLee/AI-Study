# 필요한 라이브러리 가져오기
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import streamlit as st
import plotly.express as px

def get_naver_news():
    """네이버 뉴스 IT/과학 섹션에서 기사 정보를 가져오는 함수"""

    print("📰 네이버 뉴스 IT/과학 섹션에서 기사를 가져옵니다...")

    # 네이버 뉴스 IT/과학 섹션 URL
    url = "https://news.naver.com/section/105"

    # 웹 브라우저처럼 위장하기
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        print(" 네이버 뉴스 페이지에 접속 중...")
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            print(" 네이버 뉴스 접속 성공!")
        else:
            print(f" 네이버 뉴스 접속 실패 (HTTP 상태코드: {response.status_code})")
            return None

        # HTML 분석
        soup = BeautifulSoup(response.text, 'html.parser')
        print(" 뉴스 페이지 내용을 분석 중...")

        # 뉴스 기사 정보 추출
        news_data = extract_news_articles(soup)

        if news_data and len(news_data) > 0:
            print(f" {len(news_data)}개의 기사를 성공적으로 수집했습니다!")
            return news_data
        else:
            print(" 기사 정보를 찾을 수 없습니다.")
            print("   - 네이버 뉴스 페이지 구조가 변경되었을 수 있습니다.")
            print("   - 네트워크 연결을 확인해주세요.")
            return None

    except requests.exceptions.Timeout:
        print(" 네이버 뉴스 접속 시간 초과")
        print("   - 네트워크 연결이 느리거나 불안정합니다.")
        return None
    except requests.exceptions.ConnectionError:
        print(" 네이버 뉴스 연결 오류")
        print("   - 인터넷 연결을 확인해주세요.")
        return None
    except Exception as e:
        print(f" 예상치 못한 오류 발생: {e}")
        print("   - 나중에 다시 시도해주세요.")
        return None
    
def extract_news_articles(soup):
    """웹페이지에서 뉴스 기사 정보 추출"""

    print(" 뉴스 기사를 찾는 중...")

    news_list = []

    try:
        # 네이버 뉴스의 다양한 기사 선택자들 시도
        article_selectors = [
            # 2024년 기준 네이버 뉴스 선택자들
            'div.sa_item',                          # 섹션 기사 아이템
            'div.sa_item_flex',                     # 플렉스 기사 아이템
            'li.sa_item',                           # 리스트 형태 기사
            'div.section_article',                   # 섹션 기사
            'div.news_area',                        # 뉴스 영역
            'div.list_body li',                     # 리스트 본문의 항목들
            'div.cluster_body li',                  # 클러스터 본문 항목들
            'article',                              # article 태그
            '.sa_item_flex',                        # CSS 클래스
        ]

        found_articles = []

        # 각 선택자로 기사 찾기 시도
        for i, selector in enumerate(article_selectors):
            print(f"   선택자 {i+1}: '{selector}' 시도 중...")
            articles = soup.select(selector)

            if articles:
                print(f"    '{selector}'로 {len(articles)}개 요소 발견!")
                found_articles = articles
                break
            else:
                print(f"    '{selector}'로 요소를 찾지 못함")

        # 모든 선택자 실패시 기사 링크 직접 검색
        if not found_articles:
            print(" 다른 방법: 모든 링크에서 기사 링크 찾는 중...")
            all_links = soup.find_all('a', href=True)
            article_links = []

            for link in all_links:
                href = link.get('href', '')
                if '/article/' in href and link.get_text().strip():
                    article_links.append(link)

            if article_links:
                print(f"    {len(article_links)}개의 기사 링크 발견!")
                found_articles = article_links
            else:
                print("    기사 링크를 찾지 못함")

        # 디버깅: 페이지 구조 분석
        if not found_articles:
            print("\n 페이지 구조 분석 중...")
            print(f"   전체 텍스트 길이: {len(soup.get_text())}")

            # 주요 div 클래스들 찾기
            divs_with_class = soup.find_all('div', class_=True)
            class_names = set()
            for div in divs_with_class[:20]:  # 상위 20개만
                classes = div.get('class', [])
                for cls in classes:
                    if 'sa_' in cls or 'news' in cls or 'article' in cls or 'section' in cls:
                        class_names.add(cls)

            if class_names:
                print(f"   발견된 관련 클래스: {list(class_names)[:10]}")

            return []

        # 기사 정보 추출
        print(f"📝 {len(found_articles)}개 요소에서 기사 정보 추출 중...")

        for i, article in enumerate(found_articles[:30]):  # 최대 30개
            try:
                news_info = extract_article_info(article, i + 1)
                if news_info:
                    news_list.append(news_info)
                    print(f"    {i+1}번째 기사: {news_info['제목'][:30]}...")

                # 서버 부하 방지를 위한 딜레이
                if i > 0 and i % 10 == 0:
                    time.sleep(0.3)

            except Exception as e:
                print(f"    {i+1}번째 기사 처리 중 오류: {e}")
                continue

        return news_list

    except Exception as e:
        print(f" 기사 추출 중 오류: {e}")
        return []    

def extract_article_info(article, index):
    """개별 기사에서 정보 추출"""

    try:
        # 기사 제목 찾기
        title = ""
        title_selectors = [
            'a.sa_text_title',           # 네이버 뉴스 제목 링크
            '.sa_text_title',            # 제목 클래스
            '.sa_text_strong',           # 강조 텍스트
            'strong.sa_text_strong',     # 강조 제목
            'a[href*="/article/"]',      # 기사 링크
            'a',                         # 일반 링크
            'strong',                    # 강조 태그
        ]

        for selector in title_selectors:
            title_element = article.select_one(selector)
            if title_element:
                title = title_element.get_text().strip()
                if len(title) > 10 and len(title) < 200:  # 적절한 제목 길이
                    print(f"    제목 찾음: {title[:30]}...")
                    print(f"    선택자: {selector}")
                    break

        # 제목이 없으면 article 자체가 링크인 경우 체크
        if not title and article.name == 'a':
            title = article.get_text().strip()

        # 기사 링크 찾기
        link = ""
        if article.name == 'a' and article.get('href'):
            link = article.get('href')
        else:
            link_element = article.find('a', href=lambda x: x and '/article/' in x)
            if link_element:
                link = link_element.get('href')

        # 상대 링크를 절대 링크로 변환
        if link and link.startswith('/'):
            link = 'https://news.naver.com' + link
        elif link and not link.startswith('http'):
            link = 'https://news.naver.com' + link

        # 언론사 정보 찾기
        press = ""
        press_selectors = [
            '.sa_text_press',            # 언론사 클래스
            '.press',                    # 일반 언론사
            '.source',                   # 출처
            '.sa_text_info_left',        # 왼쪽 정보
            '.byline',                   # 바이라인
        ]

        for selector in press_selectors:
            press_element = article.select_one(selector)
            if press_element:
                press = press_element.get_text().strip()
                if press and len(press) < 50:  # 적절한 언론사명 길이
                    break

        # 시간 정보 찾기
        time_info = ""
        time_selectors = [
            '.sa_text_datetime',         # 시간 클래스
            '.time',                     # 시간
            '.date',                     # 날짜
            '.sa_text_info_right',       # 오른쪽 정보
            'time',                      # time 태그
        ]

        for selector in time_selectors:
            time_element = article.select_one(selector)
            if time_element:
                time_info = time_element.get_text().strip()
                if time_info and ('시간' in time_info or '분' in time_info or '일' in time_info or ':' in time_info):
                    break

        # 기사 요약이나 부제목 찾기
        summary = ""
        summary_selectors = [
            '.sa_text_lede',             # 리드 문장
            '.summary',                  # 요약
            '.sub_title',                # 부제목
            '.description',              # 설명
        ]

        for selector in summary_selectors:
            summary_element = article.select_one(selector)
            if summary_element:
                summary = summary_element.get_text().strip()
                if summary and len(summary) > 10:
                    break

        # 유효한 기사인지 확인
        if title and len(title) > 5 and '광고' not in title:
            return {
                '순번': index,
                '제목': title[:150],  # 제목 길이 제한
                '언론사': press if press else '정보없음',
                '시간': time_info if time_info else '정보없음',
                '요약': summary[:200] if summary else '요약없음',
                '링크': link if link else '링크없음'
            }

        print("not valid article:", title)
        return None

    except Exception as e:
        return None


def save_news_data(news_data):
    """뉴스 데이터를 파일로 저장"""

    if not news_data:
        print(" 저장할 뉴스 데이터가 없습니다")
        return

    print(f"\n {len(news_data)}개 뉴스 기사를 파일로 저장합니다...")

    try:
        # DataFrame 생성
        df = pd.DataFrame(news_data)

        # CSV 파일 저장
        df.to_csv('naver_news_it.csv', index=False, encoding='utf-8')

        # 텍스트 파일 저장
        with open('naver_news_it.txt', 'w', encoding='utf-8') as f:
            f.write("=== 네이버 뉴스 IT/과학 섹션 ===\n")
            f.write(f"수집 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 기사 수: {len(news_data)}개\n")
            f.write("=" * 60 + "\n\n")

            for article in news_data:
                f.write(f"[{article['순번']}] {article['제목']}\n")
                f.write(f"언론사: {article['언론사']} | 시간: {article['시간']}\n")
                if article['요약'] != '요약없음':
                    f.write(f"요약: {article['요약']}\n")
                f.write(f"링크: {article['링크']}\n")
                f.write("-" * 50 + "\n\n")

        print(" 파일 저장 완료!")
        print("   - naver_news_it.csv (엑셀용)")
        print("   - naver_news_it.txt (텍스트용)")

        print(df)

    except Exception as e:
        print(f" 파일 저장 중 오류: {e}")


def show_news_data(news_data):
    """뉴스 데이터를 화면에 출력"""

    if not news_data:
        print(" 출력할 뉴스 데이터가 없습니다")
        return

    print(f"\n" + "="*80)
    print("  네이버 뉴스 IT/과학 섹션 - 최신 기사  ")
    print(f"수집 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 {len(news_data)}개 기사")
    print("="*80)

    for article in news_data:
        print(f"\n[{article['순번']}] {article['제목']}")
        print(f"    {article['언론사']} |  {article['시간']}")

        if article['요약'] != '요약없음' and len(article['요약']) > 10:
            print(f"    {article['요약']}")

        if article['링크'] != '링크없음':
            print(f"    {article['링크']}")

        print("-" * 70)

    print("\n" + "="*80)

def analyze_news_data(news_data):
    """뉴스 데이터 간단 분석"""

    if not news_data:
        return

    print(f"\n 뉴스 데이터 분석 결과:")

    # 언론사별 기사 수
    press_count = {}
    for article in news_data:
        press = article.get('언론사', '알수없음')
        if press != '정보없음':
            press_count[press] = press_count.get(press, 0) + 1

    if press_count:
        print(f"    참여 언론사: {len(press_count)}개")
        top_press = sorted(press_count.items(), key=lambda x: x[1], reverse=True)[:5]
        for press, count in top_press:
            print(f"      - {press}: {count}개 기사")

    # 시간 정보가 있는 기사
    time_articles = [a for a in news_data if a.get('시간', '정보없음') != '정보없음']
    print(f"    시간 정보 있는 기사: {len(time_articles)}개")

    # 요약이 있는 기사
    summary_articles = [a for a in news_data if a.get('요약', '요약없음') != '요약없음']
    print(f"    요약이 있는 기사: {len(summary_articles)}개")

# 메인 실행 함수
def main():
    """프로그램의 메인 함수"""

    print(" 네이버 뉴스 IT/과학 크롤링을 시작합니다!")
    print("=" * 60)
    print(" 대상 사이트: https://news.naver.com/section/105")
    print(" 목표: 실제 IT/과학 뉴스 기사 수집")
    print("=" * 60)

    # 뉴스 데이터 크롤링
    news_data = get_naver_news()

    if news_data:
        # 데이터 분석
        analyze_news_data(news_data)

        # 데이터 화면 출력
        show_news_data(news_data)

        # 데이터 파일 저장
        save_news_data(news_data)

        print(f"\n 뉴스 크롤링 완료!")
        print(f" 총 {len(news_data)}개의 IT/과학 뉴스를 수집했습니다!")

    else:
        print("\n 뉴스 크롤링 실패!")


# 간단한 테스트 함수
def test_news_crawling():
    """뉴스 크롤링 테스트"""

    print(" 네이버 뉴스 크롤링을 테스트합니다...\n")

    news_data = get_naver_news()

    if news_data:
        print(" 크롤링 테스트 성공!")
        print(f" 수집된 기사 수: {len(news_data)}")
        print(" 첫 3개 기사 제목:")
        for i, article in enumerate(news_data[:3]):
            print(f"   {i+1}. {article['제목'][:50]}...")
    else:
        print(" 크롤링 테스트 실패!")

    return news_data

# 프로그램 실행
if __name__ == "__main__":
    main()
