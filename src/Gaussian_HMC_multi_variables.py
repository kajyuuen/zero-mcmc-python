import numpy as np

n_iter = 10**5 # サンプル数
n_tau = 20 # リープフロッグのステップ数
d_tau = 1/2 # リープフロッグのステップ幅
n_dim = 3 # 変数の個数

# ボックス・ミューラー法によるガウス乱数の生成
def box_muller():
    # r, s ~ Uniform(0, 1)
    r, s = np.random.rand(), np.random.rand()

    # p, q ~ N(0, 1)
    p = np.sqrt(-2*np.log(r))*np.sin(2*np.pi*s)
    q = np.sqrt(-2*np.log(r))*np.cos(2*np.pi*s)

    return p, q

# 作用S(x)の計算
def  calc_action(x, A):
    action = 0
    for i in range(n_dim):
        for j in range(n_dim):
            action += x[i]*A[i][j]*x[j]
        action += 1/2 * x[i] * A[i][i] * x[i]
    return action

# ハミルトニアンH(x, p)の計算
def calc_hamiltonian(x, p, A):
    # 作用(ポテンシャルエネルギー)の計算
    h = calc_action(x, A)
    # 運動エネルギーを加えて、ハミルトニアンを計算する
    for i in range(n_dim):
        h = h + (1/2)*(p[i]**2)
    return h

# ハミルトニアンのx微分、dH/dxの計算
def calc_delh(x, A):
    delh = np.zeros(n_dim)
    
    for i in range(n_dim):
        for j in range(n_dim):
            delh[i] = delh[i] + A[i][j]*x[j]

    return delh

# リープフロッグ(分子軌道法)による時間発展
def molucular_dynamics(x, A):
    # 運動量pをガウス分布に従う乱数として生成
    p = []
    for i in range(n_dim):    
        r1, _ = box_muller()
        p.append(r1)
    p = np.array(p)

    # ハミルトニアンの計算
    ham_init = calc_hamiltonian(x, p, A)
    
    # リープフロッグの1ステップ目
    x = x + p * (1/2) * d_tau
    
    # リープフロッグの2, ..., n_tauステップ目
    for step  in range(1, n_tau):
        delh = calc_delh(x, A)
        p = p - delh * d_tau
        x = x + p * d_tau

    # リープフロッグの最終ステップ
    delh = calc_delh(x, A)
    p = p - delh * d_tau
    x = x + p * (1/2) * d_tau
    
    # ハミルトニアンの計算
    ham_fin = calc_hamiltonian(x, p, A)

    return x, ham_init, ham_fin

def main():
    # 初期配位
    x = np.zeros(n_dim)
    A = np.zeros((n_dim, n_dim))

    A[0][0], A[1][1], A[2][2] = 1, 2, 2
    A[0][1], A[0][2], A[1][2] = 1, 1, 1

    for i in range(1, n_dim):
        for j in range(n_dim):
            A[i][j] = A[j][i]

    # 更新回数
    n_accept = 0
    
    for i in range(n_iter):
        backup_x = np.copy(x)

        # リープフロッグ
        x, ham_init, ham_fin = molucular_dynamics(x, A)
        
        # メトロポリステスト
        metropolis = np.random.rand()
        if np.exp(ham_init-ham_fin) > metropolis:
            # 受理
            n_accept+= 1
        else:
            # 棄却
            x = backup_x

        if (i+1) % 10 == 0:
            print("{} {} {} {}".format(x[0], x[1], x[2], n_accept/(i+1)))

if __name__ == "__main__":
    main()