
import requests
from bs4 import BeautifulSoup

def basic():
    # 간단한 HTML 문서 예제
    html_content1 = """
    <html>
    <body>
    <h1>Welcome to HTML Heading Exploration</h1>
    <p>Notice how heading sizes change from h1 to h6:</p>

    <h1>Heading 1 - Largest and Most Important</h1>
    <p>This is an h1 heading, typically used for main titles.</p>

    <h2>Heading 2 - Section Title</h2>
    <p>H2 is used for major sections within the document.</p>

    <h3>Heading 3 - Subsection Title</h3>
    <p>H3 represents subsections or smaller divisions.</p>

    <h4>Heading 4 - Minor Heading</h4>
    <p>H4 is used for less significant headings.</p>

    <h5>Heading 5 - Very Small Heading</h5>
    <p>H5 is rarely used but available for additional hierarchy.</p>

    <h6>Heading 6 - Smallest Heading</h6>
    <p>H6 is the least prominent heading tag.</p>
    </body>
    </html>

    """

    html_content = """
    <html>
        <head>
            <title>파이썬 크롤링 연습</title>
        </head>
        <body>
            <h1>안녕하세요! 크롤링 세계에 오신 것을 환영합니다</h1>
            <div class="content">
                <p class="intro">크롤링은 웹페이지에서 데이터를 추출하는 기술입니다.</p>
                <ul>
                    <li>파이썬</li>
                    <li>자바스크립트</li>
                    <li>HTML/CSS</li>
                </ul>
            </div>
            <div class="footer">
                <p>즐거운 코딩 되세요! </p>
            </div>
        </body>
    </html>
    """

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content1, 'html.parser')

    print("1 제목 추출:")
    title = soup.find('title').text
    print(f"    제목: {title}")

    print("\n2 h1 태그 내용 추출:")
    h1_text = soup.find('h1').text
    print(f"    헤더: {h1_text}")

    print("\n3 클래스로 요소 찾기:")
    intro = soup.find('p', class_='intro').text
    print(f"    소개: {intro}")

    print("\n4 리스트 항목들 추출:")
    list_items = soup.find_all('li')
    print("    프로그래밍 언어들:")
    for i, item in enumerate(list_items, 1):
        print(f"      {i}. {item.text}")

    print("\n5 모든 p 태그 찾기:")
    all_paragraphs = soup.find_all('p')
    for i, p in enumerate(all_paragraphs, 1):
        print(f"    문단 {i}: {p.text}")


def crawl_web_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"웹 페이지를 가져오는 중 오류 발생: {e}")
        return None
    
def crawlTest():
    try:
        # 간단한 웹페이지 크롤링 (httpbin.org 사용)
        url = "https://httpbin.org/html"
        #url = "https://playdata.io/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        print("1 웹페이지 접근 중...")
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'  # 한글 인코딩 설정

        soup = BeautifulSoup(response.text, 'html.parser')

        print("1 h1 추출:")
        h1title = soup.find('h1').text
        print(f"    h1: {h1title}")

        print("2 웹페이지 제목 추출:")
        title = soup.find('title')
        if title:
            print(f"    제목: {title.text}")

        print("\n3 모든 링크 추출:")
        links = soup.find_all('a', href=True)
        for i, link in enumerate(links[:3], 1):  # 상위 3개만 출력
            print(f"   {i}. {link.text.strip()} -> {link['href']}")

        print("\n4 모든 문단 텍스트 추출:")
        paragraphs = soup.find_all('p')
        for i, p in enumerate(paragraphs[:3], 1):  # 상위 3개만 출력
            text = p.text.strip()
            if text:
                print(f"    문단 {i}: {text}")

        print("\n 크롤링 완료!")

    except Exception as e:
        print(f" 오류가 발생했습니다: {e}")
        print(" 팁: 인터넷 연결을 확인해보세요.")

        # 예시 데이터로 대체
        print("\n 예시 데이터로 크롤링 시뮬레이션:")
        sample_data = [
            "파이썬으로 웹 크롤링하기",
            "BeautifulSoup 사용법 배우기",
            "데이터 수집과 분석의 첫걸음"
        ]

        for i, item in enumerate(sample_data, 1):
            print(f"   {i}. {item}")

if __name__ == "__main__":
    crawlTest()   