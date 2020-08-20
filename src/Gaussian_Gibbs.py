import numpy as np

n_iter = 10**4 # サンプル数

# ボックス・ミューラー法によるガウス乱数の生成
def box_muller():
    # r, s ~ Uniform(0, 1)
    r, s = np.random.rand(), np.random.rand()

    # p, q ~ N(0, 1)
    p = np.sqrt(-2*np.log(r))*np.sin(2*np.pi*s)
    q = np.sqrt(-2*np.log(r))*np.cos(2*np.pi*s)

    return p, q

def main():
    A = np.zeros((3, 3))

    A[0][0], A[1][1], A[2][2] = 1, 2, 2
    A[0][1], A[0][2], A[1][2] = 1, 1, 1
    A[1][0] = A[0][1]
    A[2][0] = A[0][2]
    A[2][1] = A[1][2]

    # 初期値の設定
    x, y, z = 0, 0, 0

    for i in range(n_iter):
        # xの更新
        sigma = 1/np.sqrt(A[0][0])
        mu = -A[0][1]/A[0][0]*y - A[0][2]/A[0][0]*z
        r1, _ = box_muller()
        x = sigma*r1+mu

        # yの更新
        sigma = 1/np.sqrt(A[1][1])
        mu = -A[1][0]/A[1][1]*x - A[1][2]/A[1][1]*z
        r1, _ = box_muller()
        y = sigma*r1+mu

        # zの更新
        sigma = 1/np.sqrt(A[2][2])
        mu = -A[2][0]/A[2][2]*x - A[2][1]/A[2][2]*y
        r1, _ = box_muller()
        z = sigma*r1+mu
        
        if (i+1) % 10 == 0:
            print("{} {} {}".format(x, y, z))

if __name__ == "__main__":
    main()