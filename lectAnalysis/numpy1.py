
import numpy as np

def test1():
    a = [1, 2, 3, 4, 5]
    b = [6, 7, 8, 9, 10]
    # print("원본 배열:", a[1])
    # for i in range(0, len(a)):
    #     print(f"i = {a[i]}")

    # zip >> ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    dap = 0
    for x, y in zip(a, b):
        dap = x + y
        print(f"x = {x}, y = {y}, dap = {dap}")

def test2():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    # print("원본 배열:", a[1])
    # for i in range(0, len(a)):
    #     print(f"i = {a[i]}")

    # zip >> ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    dap = a + b
    print(f"a + b = {dap}") 
    print(f"a * b = {a * b}") 
              
def test3():
    data = np.array([10, 20, 30, 40, 50])        

    print("numpy + 5:", data + 5)     
    print("mean:", data.mean())   
    print("mean2:", np.mean(data))     
    print("average:", np.average(data))     

def test4():
    arr = np.array([1, 2, 3, 4, 5])
    print("shape:", np.shape(arr))  # 배열의 형태 출력

    print(np.zeros(5))                # 출력 : [0. 0. 0. 0. 0.] → 0 으로 채워진 1차원 배열 생성
    print(np.ones((5, 2)))                 # 출력 : [1. 1. 1. 1. 1.] → 1 로 채원진 1차원 배열 생성
    print(np.empty(5))                # 출력 : [1. 1. 1. 1. 1.] → 빈 배열 생성, 메모리 상태에 따라 값이 달라진다.
    print(np.full((5, 3), 7))              # 출력 : [7 7 7 7 7] →7 로 5 개 채워진 1차원 배열 생성
    print(np.arange(5))               # 출력 : [0 1 2 3 4] → 0 부터 시작하는 5 개의 숫자 생성 (0, 1, 2, 3, 4)
    print(np.arange(2,20,3))          # 출력 : [ 2  5  8 11 14 17] → 일정한 간격의 배열 생성, (시작, 정지, 단계)
    print(np.random.random(3))        # 출력 : 0 이상 1 미만의 난수 3개를 랜덤하게 배열 생성
    print(np.linspace(0,100, num=3))  # 출력 : [  0.  50. 100.] → 선형 간격으로 배치된 값으로 배열 생성

def test5():
    print(np.zeros((2, 3)))             # 출력 : 2행 3열의 0으로 채워진 배열 생성
    print(np.ones((2, 3)))              # 출력 : 2행 3열의 1로 채워진 배열 생성
    print(np.empty((2, 3)))             # 출력 : 2행 3열의 "초기화되지 않은" 배열 (값은 메모리에 따라 달라짐)
    print(np.full((2, 3), 7))           # 출력 : 2행 3열이 모두 7로 채워진 배열 생성
    print(np.arange(6).reshape(2, 3))   # 출력 : 0부터 시작하는 6개의 정수를 2행 3열로 재배열
    print(np.arange(2, 20, 3).reshape(2, 3))         # 출력 : 2부터 시작해서 3씩 증가하는 수를 2행 3열 배열로 구성
    print(np.random.random((2, 3)))     # 출력 : 2행 3열의 난수를 0 이상 1 미만 범위로 랜덤 생성
    print(np.linspace(0, 100, num=6).reshape(2, 3))  # 출력 : 0부터 100까지 균등하게 나눈 6개 숫자를 2행 3열로 재배열

def test6():
    arr = np.array([[1, 2, 3], [4, 5, 6]])


    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    arr = arr.reshape(-1, 2)  # 2행 5열로 재배열
    # print("원본 배열:\n", arr)  
    # print("배열의 차원:", arr.ndim)   # 배열의 차원

    img_rd_data = np.arange(1, 64)
    rgb_data = img_rd_data.reshape(-1, 3)  # -1행 3열로 재배열
    print("RGB 데이터:\n", rgb_data)

def test7():
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    dap = arr.transpose()
    print(dap)

if __name__ == "__main__":
    test7()
