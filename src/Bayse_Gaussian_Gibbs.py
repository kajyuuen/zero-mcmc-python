import numpy as np

# イテレーション回数
n_iter = 1000

# n個のサンプル {x, y} = {{x_1, y_1}, ... ,{x_n, y_n}}
n_sample = 100

# n個のサンプルx, yを用いた平均
av_x = -0.0930181
av_y = 0.0475899
av_xx = 1.06614
av_yy = 1.28152
av_xy = -0.504944

# step
step_A = 0.1
step_mu = 0.1

n_skip = 10

# ボックス・ミューラー法によるガウス乱数の生成
def box_muller():
    # r, s ~ Uniform(0, 1)
    r, s = np.random.rand(), np.random.rand()

    # p, q ~ N(0, 1)
    p = np.sqrt(-2*np.log(r))*np.sin(2*np.pi*s)
    q = np.sqrt(-2*np.log(r))*np.cos(2*np.pi*s)

    return p, q

# 作用の計算
def calc_action(A, mu):
    action =  A[0][0] * ((mu[0]-av_x) * (mu[0]-av_x) + av_xx- av_x * av_x) \
        + A[1][1] * ((mu[1]-av_y) * (mu[1]-av_y) + av_yy - av_y * av_y) \
        + A[0][1] * ((mu[0]-av_x) * (mu[1]-av_y) + av_xy - av_x * av_y) * 2 \
        - np.log(A[0][0] * A[1][1]-A[0][1] * A[1][0])

    action = (1/2) * action * n_sample
  
    action = action \
            + (1/2) * A[0][0] * A[0][0] \
            + (1/2) * A[0][0] * A[1][1] \
            + A[0][1] * A[0][1] \
            + (1/2) * mu[0] * mu[0] \
            + (1/2) * mu[1] * mu[1]
    
    return action

def main():
    A = np.zeros((2, 2))

    A[0][0], A[1][1] = 1, 1
    A[0][1] = 0
    A[1][0] = A[0][1]

    mu = np.zeros(2)

    n_accept = 0
    for i in range(n_iter):
        # muのギブスサンプリング
        r1, r2 = box_muller()
        mu[0] = r1 / np.sqrt(1 + n_sample * A[0][0])
        mu[0] = mu[0] - n_sample / (1 + n_sample * A[0][0]) * (A[0][1]* (mu[1] - av_y) - A[0][0] * av_x)

        mu[1] = r2 / np.sqrt(1 + n_sample * A[1][1])
        mu[1] = mu[1] - n_sample / (1 + n_sample * A[1][1]) * (A[1][0]* (mu[0] - av_x) - A[1][1] * av_x)

        # Aの候補をメトロポリス法によって選択する
        backup_A = np.copy(A)

        action_init = calc_action(A, mu)

        # 各パラメータの候補を選ぶ
        # Aについて
        dx = np.random.rand() - 1/2
        A[0][0] += dx * step_A * 2
        dx = np.random.rand() - 1/2
        A[1][1] += dx * step_A * 2
        dx = np.random.rand() - 1/2
        A[1][0] += dx * step_A * 2
        A[0][1] += dx * step_A * 2

        action_fin = calc_action(A, mu)

        # メトロポリステスト
        metropolis = np.random.rand()
        if np.exp(action_init - action_fin) > metropolis:
            # 受理
            n_accept += 1
        else:
            # 棄却
            A = np.copy(backup_A)

        if (i+1) % n_skip == 0:
            print("{}\n{}\n{}\n".format(A, mu, n_accept/(i+1)))


if __name__ == "__main__":
    main()