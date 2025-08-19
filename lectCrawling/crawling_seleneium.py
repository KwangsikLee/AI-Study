from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

import re
import pandas as pd
import time


def setup_driver():
    # 크롬 설정 (headless 환경 가능)
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    # chrome_options.binary_location = "/usr/bin/google-chrome"

    # # 드라이버 실행
    # service = Service("/usr/bin/chromedriver")
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    driver = webdriver.Chrome()


    # 멜론 연도별 차트 URL
    url = "https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2000"
    driver.get(url)
    time.sleep(5)  # JS 렌더링 대기

    # 전체 페이지 소스를 문자열로 가져오기
    html = driver.page_source

    # ✅ 정규표현식으로 goAlbumDetail('123456') 형태 찾기
    #album_ids = re.findall(r"goAlbumDetail\('(\d+)'\)", html)
    album_ids = list(set(re.findall(r"goAlbumDetail\('(\d+)'\)", html)))  # 중복 제거


    print("페이지 내 albumDetail 함수 개수:", len(re.findall(r"goAlbumDetail\('(\d+)'\)", html)))


    # ✅ URL 생성
    album_urls = [f"https://www.melon.com/album/detail.htm?albumId={aid}" for aid in album_ids]

    # ✅ 결과 출력 및 저장
    df = pd.DataFrame({
        'album_id': album_ids,
        'album_url': album_urls
    })
    print(df)
    df.to_csv("melon_album_urls_1994.csv", index=False)

    driver.quit()

def get_album_urls():
    # ▶ 1. 앨범 URL 목록 로드
    album_df = pd.read_csv("melon_album_urls_1994.csv")
    album_urls = album_df['album_url'].tolist()

    # ▶ 2. 크롬 드라이버 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    chrome_options.binary_location = "/usr/bin/google-chrome"

    # service = Service("/usr/bin/chromedriver")
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    driver = webdriver.Chrome()

    # ▶ 3. 결과 저장 리스트
    result = []

    for album_url in album_urls:
        try:
            driver.get(album_url)
            time.sleep(2)

            # 앨범명과 아티스트명
            album_title = driver.find_element(By.CLASS_NAME, "song_name").text.replace("앨범명", "").strip()
            artist_name = driver.find_element(By.CLASS_NAME, "artist").text.strip()

            # 수록곡 리스트
            song_rows = driver.find_elements(By.CSS_SELECTOR, 'div#d_song_list table > tbody > tr')

            print(f"📀 {album_title} - {artist_name} 앨범 수록곡 수: {len(song_rows)}")

            for row in song_rows:
                try:
                    title_tag = row.find_element(By.CSS_SELECTOR, 'div.ellipsis.rank01 a')
                    song_title = title_tag.text.strip()
                    song_href = title_tag.get_attribute('href')

                    # 곡 상세 페이지로 이동
                    driver.get(song_href)
                    time.sleep(2)

                    # 가사 추출
                    lyrics_element = driver.find_element(By.CSS_SELECTOR, 'div.lyric')
                    lyrics = lyrics_element.text.strip().replace('\n', ' ')

                    print(f"🎵 {album_title} - {artist_name} - {song_title} 가사 수집 완료")
                    result.append({
                        "album": album_title,
                        "artist": artist_name,
                        "title": song_title,
                        "lyrics": lyrics
                    })

                    driver.back()
                    time.sleep(1)
                except Exception as e:
                    print(f"⚠️ 곡 처리 실패: {e}")
                    continue
        except Exception as e:
            print(f"❌ 앨범 처리 실패: {e}")
            continue

    driver.quit()

    # ▶ 4. 결과 저장
    df_result = pd.DataFrame(result)
    df_result.to_csv("melon_lyrics_1994.csv", index=False)
    print("✅ melon_lyrics_1994.csv 저장 완료")


def get_album_urls_and_lyrics():
    # ▶ 2. 크롬 드라이버 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    chrome_options.binary_location = "/usr/bin/google-chrome"

    # service = Service("/usr/bin/chromedriver")
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    driver = webdriver.Chrome()

    # ▶ 대상 앨범 URL
    album_url = "https://www.melon.com/album/detail.htm?albumId=3933"
    album_id = re.search(r'albumId=(\d+)', album_url).group(1)

    driver.get(album_url)
    time.sleep(3)

    # ▶ 앨범명 / 아티스트명 추출
    album_title = driver.find_element(By.CLASS_NAME, "song_name").text.replace("앨범명", "").strip()
    artist_name = driver.find_element(By.CLASS_NAME, "artist").text.strip()

    # ▶ 곡 ID 추출 (goSongDetail)
    html = driver.page_source
    song_ids = re.findall(r"goSongDetail\('(\d+)'\)", html)
    print(f"🎶 추출된 곡 수: {len(song_ids)}")

    # ▶ 곡별 수집
    result = []
    for song_id in song_ids:
        try:
            song_url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
            driver.get(song_url)
            time.sleep(2)

            # 곡 제목
            title = driver.find_element(By.CSS_SELECTOR, 'div.song_name').text.replace("곡명", "").strip()

            # 가사
            lyrics_tags = driver.find_elements(By.CSS_SELECTOR, 'div.lyric')
            lyrics = lyrics_tags[0].text.strip().replace('\n', ' ') if lyrics_tags else ""

            result.append({
                "albumid": album_id,
                "songid": song_id,
                "album": album_title,
                "artist": artist_name,
                "title": title,
                "lyrics": lyrics
            })
        except Exception as e:
            print(f"⚠️ 오류 (songId={song_id}):", e)
            continue

    driver.quit()

    # ▶ CSV 저장
    df = pd.DataFrame(result)
    df.to_csv("melon_lyrics_3933.csv", index=False)
    print("✅ melon_lyrics_3933.csv 저장 완료!")


def main():
    setup_driver()
    get_album_urls()


if __name__ == "__main__":
    main()