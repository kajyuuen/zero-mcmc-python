import math
from random import random

n_iter = 10**4 # サンプル数
n_tau = 10 # リープフロッグのステップ数
d_tau = 0.1 # リープフロッグのステップ幅

# ボックス・ミューラー法によるガウス乱数の生成
def box_muller():
    # r, s ~ Uniform(0, 1)
    r, s = random(), random()

    # p, q ~ N(0, 1)
    p = math.sqrt(-2*math.log(r))*math.sin(2*math.pi*s)
    q = math.sqrt(-2*math.log(r))*math.cos(2*math.pi*s)

    return p, q

# 作用S(x)の計算
def  calc_action(x):
    action = 1/2*(x**2)
    return action

# ハミルトニアンH(x, p)の計算
def calc_hamiltonian(x, p):
    # 作用(ポテンシャルエネルギー)の計算
    s = calc_action(x)
    # 運動エネルギーを加えて、ハミルトニアンを計算する
    h = s + (1/2)*(p**2)
    return h

# ハミルトニアンのx微分、dH/dxの計算
def calc_delh(x):
    # S(x)' = (1/2*x^2)' = x
    delh = x
    return delh

# リープフロッグ(分子軌道法)による時間発展
def molucular_dynamics(x):

    # 運動量pをガウス分布に従う乱数として生成
    r1, _ = box_muller()
    p = r1

    # ハミルトニアンの計算
    ham_init = calc_hamiltonian(x, p)
    
    # リープフロッグの1ステップ目
    x = x + p*(1/2)*d_tau
    
    # リープフロッグの2, ..., n_tauステップ目
    for step  in range(1, n_tau):
        delh = calc_delh(x)
        p = p - delh*d_tau
        x = x + p*d_tau
    
    # リープフロッグの最終ステップ
    delh = calc_delh(x)
    p = p - delh*d_tau
    x = x + p*(1/2)*d_tau

    # ハミルトニアンの計算
    ham_fin = calc_hamiltonian(x, p)    

    return x, ham_init, ham_fin

def main():
    # 初期配位
    x = 0

    # 更新回数
    n_accept = 0
    # x^2の和
    sum_xx = 0
    
    for i in range(n_iter):
        backup_x = x

        # リープフロッグ
        x, ham_init, ham_fin = molucular_dynamics(x)
        
        # メトロポリステスト
        metropolis = random()
        if math.exp(ham_init-ham_fin) > metropolis:
            # 受理
            n_accept+= 1
        else:
            # 棄却
            x = backup_x

        # データ出力
        sum_xx += x**2
        
        print('{} {} {}'.format(x, sum_xx/(i+1), n_accept/(i+1)))

if __name__ == "__main__":
    main()