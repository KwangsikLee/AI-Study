import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk


def calulateView():
    col1,col2,col3 = st.columns([1,1,1])
    # 공간을 2:3 으로 분할하여 col1과 col2라는 이름을 가진 컬럼을 생성합니다.  

    if 'numVal1' not in st.session_state:
        st.session_state.numVal1 = ""
    if 'numVal2' not in st.session_state:
        st.session_state.numVal2 = ""
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = 1   
    if 'numRet' not in st.session_state:
        st.session_state.numRet = ""

    def add_to_num(num):
        """숫자를 추가하는 함수"""
        if st.session_state.input_mode == 1:
            # numVal1에 숫자를 추가
            st.session_state.numVal1 += str(num)
        else:
            # numVal2에 숫자를 추가
            st.session_state.numVal2 += str(num)    
        # st.rerun()  # 필요시 화면을 새로고침

    def clear_num():
        """숫자를 초기화하는 함수"""
        st.session_state.numVal1 = ""
        st.session_state.numVal2 = ""
        st.session_state.input_mode = 1  # 초기화 시 입력 모드를 numVal1로 설정
        st.session_state.numRet = ""
        st.rerun()  # 필요시 화면을 새로고침

    num1 = ""
    with col1 :
    # column 1 에 담을 내용
        if st.button('1', key="num1", use_container_width=True) :
            add_to_num(1)        
        if st.button('4', key="num4", use_container_width=True) :
            add_to_num(4)
        if st.button('7', key="num7", use_container_width=True) :
            add_to_num(7)
        if st.button('c', use_container_width=True) :
            clear_num()
        
    with col2 :
        if st.button('2', key="num2", use_container_width=True) :
            add_to_num(2)        
        if st.button('5', key="num5", use_container_width=True) :
            add_to_num(5)
        if st.button('8', key="num8", use_container_width=True) :
            add_to_num(8)
        if st.button('0', key="num0", use_container_width=True) :
            add_to_num(0)
    with col3 :
        if st.button('3', key="num3", use_container_width=True) :
            add_to_num(3)        
        if st.button('6', key="num6", use_container_width=True) :
            add_to_num(6)
        if st.button('9', key="num9", use_container_width=True) :
            add_to_num(9)
        if st.button('next', key="numN", use_container_width=True) :
            st.session_state.input_mode = 2  # numVal2로 입력 모드 변경


    op_selected = st.selectbox(    'Select Operation',
        ['choose','+', '-', '*', '/'],
        # index=0,
        key='op_select',
        help='Select the operation you want to perform'
    )


    if st.session_state.numVal1 and st.session_state.numVal2:
        if op_selected == '+':
            st.session_state.numRet = str(int(st.session_state.numVal1) + int(st.session_state.numVal2))
        elif op_selected == '-':
            st.session_state.numRet = str(int(st.session_state.numVal1) - int(st.session_state.numVal2))
        elif op_selected == '*':
            st.session_state.numRet = str(int(st.session_state.numVal1) * int(st.session_state.numVal2))   
        elif op_selected == '/':
            if int(st.session_state.numVal2) != 0:
                st.session_state.numRet = str(int(st.session_state.numVal1) / int(st.session_state.numVal2))
            else:
                st.session_state.numRet = "Error: Division by zero"    

    st.write("Number 1:", st.session_state.numVal1 )
    st.write("Number 2:", st.session_state.numVal2 )
    st.write("Number Ret:", st.session_state.numRet )


def show_pandas():    
    # 샘플 데이터
    df = pd.DataFrame({
        "과일": ["사과", "바나나", "체리", "사과", "바나나", "체리"],
        "판매량": [10, 15, 8, 12, 18, 6],
        "지점": ["서울", "서울", "서울", "부산", "부산", "부산"]
    })

    graph_selected = st.selectbox(    'Select Graph',
        ['barV','line', 'pie'],
        # index=0,
        key='graph_selected',
        help='Select the graph you want to perform'
    )
    if graph_selected == 'barV':
        # plotly 그래프 생성
        fig = px.bar(df, x="과일", y="판매량", color="지점", barmode="group", title="과일별 판매량")
        st.plotly_chart(fig, use_container_width=True)
    elif graph_selected == 'line':
        # plotly 선 그래프 생성
        fig = px.line(df, x="과일", y="판매량", color="지점", title="과일별 판매량 추세", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    elif graph_selected == 'pie':
        # plotly 파이차트 생성
        grouped = df.groupby([ "지점"], as_index=False)["판매량"].sum()
        fig = px.pie(grouped, names="지점", values="판매량", title="과일별 총 판매량 비율", hole=0)
        st.plotly_chart(fig, use_container_width=True)


def show_map():
    # 데이터 정의 (서울 주요 지점 예시)
    data = pd.DataFrame({
        'lat': [37.5665, 37.5700, 37.5796],
        'lon': [126.9780, 126.9920, 126.9770],
        'place': ['시청', '동대문', '경복궁']
    })

    # pydeck으로 고급 지도 시각화
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',  # 지도 스타일
        initial_view_state=pdk.ViewState(
            latitude=37.5665,     # 초기 중심 위도
            longitude=126.9780,   # 초기 중심 경도
            zoom=11,              # 줌 레벨
            pitch=45              # 기울기
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=data,
                get_position='[lon, lat]',  # 열 이름 주의!
                get_color='[200, 30, 0, 160]',  # 빨간색 마커
                get_radius=300,  # 반경
                pickable=True,
            )
        ],
        tooltip={"text": "{place}"}
    ))


if __name__ == "__main__":
    show_pandas()