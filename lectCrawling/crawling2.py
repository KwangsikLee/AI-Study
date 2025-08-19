from bs4 import BeautifulSoup
import requests

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import matplotlib.font_manager as fm

# 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False
font_path='/System/Library/Fonts/AppleGothic.ttf'
print("1 샘플 리뷰 데이터 준비 중...")

# 영화 리뷰 샘플 데이터
sample_reviews = [
    "이 영화는 정말 재미있었습니다. 액션 장면이 훌륭하고 스토리도 좋았어요.",
    "배우들의 연기가 너무 좋았습니다. 특히 주인공의 연기가 인상적이었어요.",
    "영상미가 아름다웠고 음악도 좋았습니다. 감동적인 스토리였어요.",
    "액션 영화치고는 스토리가 탄탄했습니다. 재미있게 봤어요.",
    "연출이 뛰어나고 배우들의 케미가 좋았습니다. 추천해요.",
    "영화관에서 보길 잘했다는 생각이 듭니다. 대박 영화예요.",
    "스토리가 예측 가능했지만 그래도 재미있었습니다.",
    "액션 장면이 현실적이고 긴장감이 있었어요. 좋은 영화입니다.",
    "배우들의 연기력이 뛰어나고 연출도 훌륭했습니다.",
    "음악과 영상이 조화롭게 어우러진 작품이었어요. 감동적이었습니다.",
    "주인공의 캐릭터가 매력적이고 스토리 전개가 별로웠어요.",
    "액션 시퀀스가 박진감 넘치고 연출이 세련되었습니다.",
    "배우들의 호흡이 좋았고 대사도 자연스러웠어요.",
    "영화의 메시지가 깊이 있고 생각할 거리를 많이 주었습니다.",
    "시각적 효과가 뛰어나고 사운드도 완벽했어요."
]

print(f" 총 {len(sample_reviews)}개의 리뷰를 분석합니다.")

print("\n2 텍스트 데이터 전처리 중...")

# 모든 리뷰 텍스트 합치기
all_text = ' '.join(sample_reviews)

# 한글만 추출 (불필요한 문자 제거)
korean_text = re.sub(r'[^가-힣\s]', '', all_text)

# 단어 분리
words = korean_text.split()

# 불용어 제거 (자주 나오지만 의미없는 단어들)
stop_words = ['이', '그', '저', '것', '수', '등', '들', '의', '가', '을', '를', '에', '와', '과', '도', '은', '는', '게', '도']
meaningful_words = [word for word in words if len(word) > 1 and word not in stop_words]

print("3 자주 사용된 단어 TOP 10:")
word_count = Counter(meaningful_words)
for i, (word, count) in enumerate(word_count.most_common(10), 1):
    print(f"   {i:2d}. {word}: {count}번")

print("\n4 워드클라우드 생성 중...")

# 한글 폰트 경로 설정
# font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

try:
    # 워드클라우드 생성
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        max_words=50,
        colormap='viridis',
        font_path=font_path,  # 한글 폰트 경로 지정
        prefer_horizontal=0.9
    ).generate(' '.join(meaningful_words))

    # 워드클라우드 시각화
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(' 영화 리뷰 워드클라우드', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    print(" 워드클라우드 생성 완료!")

except Exception as e:
    print(f" 워드클라우드 생성 오류: {e}")
    print(" 대신 단어 빈도 그래프를 생성합니다.")

    # 대체: 막대 그래프로 단어 빈도 시각화
    top_words = word_count.most_common(10)
    words_list = [word for word, count in top_words]
    counts_list = [count for word, count in top_words]

    plt.figure(figsize=(12, 6))
    plt.bar(words_list, counts_list, color='skyblue')
    plt.title(' 자주 사용된 단어 TOP 10', fontsize=14)
    plt.xlabel('단어')
    plt.ylabel('빈도')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\n5 간단한 감정 분석:")

# 긍정적 단어들
positive_words = ['좋', '재미', '훌륭', '멋진', '최고', '대박', '추천', '감동', '완벽', '뛰어', '흥미', '매력']
positive_count = sum(1 for word in meaningful_words if any(pos in word for pos in positive_words))

# 부정적 단어들
negative_words = ['나쁘', '별로', '실망', '지루', '안좋', '최악']
negative_count = sum(1 for word in meaningful_words if any(neg in word for neg in negative_words))

print(f"    긍정적 단어: {positive_count}개")
print(f"    부정적 단어: {negative_count}개")
print(f"    전체 의미있는 단어: {len(meaningful_words)}개")
print(f"    긍정도: {positive_count/len(meaningful_words)*100:.1f}%")

# 감정 분석 결과 시각화
sentiment_data = ['긍정', '부정', '중립']
sentiment_counts = [positive_count, negative_count, len(meaningful_words) - positive_count - negative_count]

plt.figure(figsize=(8, 6))
colors = ['lightgreen', 'lightcoral', 'lightgray']
plt.pie(sentiment_counts, labels=sentiment_data, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title(' 감정 분석 결과', fontsize=14)
plt.axis('equal')
plt.show()