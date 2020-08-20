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

# tau
n_tau = 50
dtau_A = 0.05
dtau_mu= 0.05

n_skip = 10
n_dim = 2

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

# ハミルトニアンH(x, p)の計算
def calc_hamiltonian(p_A, p_mu, A, mu):
    # 作用(ポテンシャルエネルギー)の計算
    h = calc_action(A, mu)
    # 運動エネルギーを加えて、ハミルトニアンを計算する
    for i in range(n_dim):
        h = h + (1/2)*(p_mu[i]**2)
        for j in range(n_dim):
            h = h + (1/2)*p_A[i][j]*p_A[j][i]
    return h

# ハミルトニアンのx微分、dH/dxの計算
def calc_delh(A, mu):
    # Aの逆行列A^{-1}を求める
    A_inv = np.linalg.inv(A)

    # Aの微分dA/dxを求める
    delh_A = np.zeros_like(A)
    delh_A[0][0] = A[0][0] + (mu[0]**2 - 2 * av_x * mu[0] + av_xx - A_inv[0][0]) * (1/2) * n_sample
    delh_A[1][1] = A[1][1] + (mu[1]**2 - 2 * av_y * mu[1] + av_yy - A_inv[1][1]) * (1/2) * n_sample
    delh_A[0][1] = A[0][1] + (mu[0] * mu[1] - av_x * mu[1] - av_y * mu[0] + av_xy - A_inv[0][1]) * (1/2) * n_sample
    delh_A[1][0] = delh_A[0][1]

    # muの微分dmu/dxを求める
    delh_mu = np.zeros_like(mu)
    delh_mu[0] = mu[0] + (A[0][0] * (mu[0] - av_x)+ A[0][1] * (mu[1] - av_y)) * n_sample
    delh_mu[1] = mu[1] + (A[1][0] * (mu[0] - av_x)+ A[1][1] * (mu[1] - av_y)) * n_sample

    return delh_A, delh_mu

# リープフロッグ(分子軌道法)による時間発展
def molucular_dynamics(A, mu):

    # 運動量pをガウス分布に従う乱数として生成
    p_mu = np.zeros_like(mu)
    r1, r2 = box_muller()
    p_mu[0], p_mu[1] = r1, r2

    p_A = np.zeros_like(A)
    r1, r2 = box_muller()
    p_A[0][0], p_A[1][1] = r1, r2
    r1, _ = box_muller()
    p_A[0][1] = np.sqrt(r1)
    p_A[1][0] = p_A[0][1]

    # ハミルトニアンの計算
    ham_init = calc_hamiltonian(p_A, p_mu, A, mu)
    
    # リープフロッグの1ステップ目
    for i in range(n_dim):
        mu[i] = mu[i] + p_mu[i] * (1/2) * dtau_mu
        for j in range(n_dim):
            A[i][j] = A[i][j] + p_A[i][j] * (1/2) * dtau_A

    # リープフロッグの2, ..., n_tauステップ目
    for step in range(1, n_tau):
        delh_A, delh_mu = calc_delh(A, mu)
        for i in range(n_dim):
            p_mu[i] = p_mu[i] - delh_mu[i] * dtau_mu
            for j in range(n_dim):
                p_A[i][j] = p_A[i][j] - delh_A[i][j] * dtau_A
        for i in range(n_dim):
            mu[i] = mu[i] + p_mu[i] * dtau_mu
            for j in range(n_dim):
                A[i][j] = A[i][j] + p_A[i][j] * dtau_A
    
    # リープフロッグの最終ステップ
    delh_A, delh_mu = calc_delh(A, mu)
    for i in range(n_dim):
        p_mu[i] = p_mu[i] - delh_mu[i] * dtau_mu
        for j in range(n_dim):
            p_A[i][j] = p_A[i][j] - delh_A[i][j] * dtau_A
    for i in range(n_dim):
        mu[i] = mu[i] + p_mu[i] * (1/2) * dtau_mu
        for j in range(n_dim):
            A[i][j] = A[i][j] + p_A[i][j] * (1/2) * dtau_A

    # ハミルトニアンの計算
    ham_fin = calc_hamiltonian(p_A, p_mu, A, mu)

    return A, mu, ham_init, ham_fin

def main():
    A = np.zeros((2, 2))

    A[0][0], A[1][1] = 1, 1
    A[0][1] = 0
    A[1][0] = A[0][1]

    mu = np.zeros(2)

    n_accept = 0
    for i in range(n_iter):
        backup_A = np.copy(A)
        backup_mu = np.copy(mu)

        # リープフロッグ
        A, mu, ham_init, ham_fin = molucular_dynamics(A, mu)

        # メトロポリステスト
        metropolis = np.random.rand()
        if np.exp(ham_init - ham_fin) > metropolis:
            # 受理
            n_accept += 1
        else:
            # 棄却
            A = np.copy(backup_A)
            mu = np.copy(backup_mu)

        if (i+1) % n_skip == 0:
            print("{}\n{}\n{}\n".format(A, mu, n_accept/(i+1)))


if __name__ == "__main__":
    main()