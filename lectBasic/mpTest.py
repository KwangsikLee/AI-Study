
import matplotlib.pyplot as mp
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 3차원 그리기 모듈


def matplot_test():
    print(matplotlib.get_backend())
    mp.plot([1, 2, 3, 4])
    mp.show()

def matplot_test2():
    mp.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bo--')
    mp.plot([1, 2, 4, 5], [1, 2, 3, 8], 'ro-')
    mp.xlabel('X-data')
    mp.ylabel('Y-data')
    mp.show()

def bar_chart():
    x = np.arange(2, 6)
    print(x)

def show_3d():
    n = 100
    # 초기값
    xmin, xmax, ymin, ymax, zmin, zmax = 0, 20, 0, 20, 0, 50
    cmin, cmax = 0, 2
    xs = (xmax - xmin) * np.random.rand(n) + xmin
    ys = (xmax - xmin) * np.random.rand(n) + ymin
    zs = (xmax - xmin) * np.random.rand(n) + zmin
    color = (xmax - xmin) * np.random.rand(n) + cmin
    fig = mp.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d') # 3차원 지정 ‘3d’
    # 3차원 출력
    ax.scatter(xs, ys, zs, c=color, marker='o', s=15, cmap='Greens')
    mp.show()

def main():
    show_3d()

if __name__ == "__main__":
    main()