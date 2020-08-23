import numpy as np

# 逆温度: レプリカの個数
n_beta = 2000
# ステップ数
step_size = 0.1
# 逆温度の感覚
d_beta = 0.5

# イテレーション回数
n_iter = 1000

# f(x) = (x-1)^2 * ((x+1)^2 + 1)
def calc_f(x):
    fx = ((x-1)**2) * ((x+1)**2 + 1)
    return fx

def main():
    n_accept = np.zeros(n_beta)
    x = np.zeros(n_beta)
    beta = np.zeros(n_beta)

    # 初期化
    for i_beta in range(n_beta):
        beta[i_beta] = (i_beta+1)*d_beta

    for i in range(1, n_iter+1):
        for i_beta in range(0, n_beta):
            backup_x_i = x[i_beta]

            # S(x) = f(x) * beta
            action_init = calc_f(x[i_beta]) * beta[i_beta]
                    
            # dx ~ Uniform(-step_size, step_size)
            dx = np.random.rand()
            dx = (dx-1/2)*step_size*2        
    
            x[i] += dx

            # S(x') = f(x') * beta
            action_fin = calc_f(x[i_beta]) * beta[i_beta]
        
            # メトロポリステスト
            metropolis = np.random.rand()
            if np.exp(action_init-action_fin) > metropolis:
                # 受理
                n_accept[i_beta] += 1
            else:
                # 棄却
                x[i_beta] = backup_x_i

        # レプリカ交換
        for i_beta in range(n_beta-1):
            current_x, current_beta = x[i_beta], beta[i_beta]
            next_x, next_beta = x[i_beta+1], beta[i_beta+1]

            action_init = calc_f(current_x) * current_beta \
                                + calc_f(next_x) * next_beta

            action_fin = calc_f(current_x) * next_beta \
                            + calc_f(next_x) * current_beta

            # メトロポリステスト
            metropolis = np.random.rand()
            if np.exp(action_init-action_fin) > metropolis:
                # 受理
                x[i_beta] = next_x
                x[i_beta+1] = current_x

    print("{}\t{}\t{}".format(x[19], x[199], x[1999]))

if __name__ == "__main__":
    main()