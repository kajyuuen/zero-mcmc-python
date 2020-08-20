import math
from random import random

def main():
    n_iter = 1000
    step_size = 0.03

    # 初期値
    p = 0.5
    n = 1000
    k = 515

    # メイン
    n_accept = 0
    for i in range(1, n_iter+1):
        backup_p = p

        # 作用S(p)
        action_init = -k*math.log(p)-(n-k)*math.log(1-p)+100*(p-9/10)

        # dp ~ Uniform(-step_size, +step_size)
        dp = random()
        dp = (dp - 1/2)*step_size*2

        p += dp

        # 作用S(p')
        action_fin = - k*math.log(p)-(n-k)*math.log(1-p)+100*(p-9/10)

        # メトロポリステスト
        metropolis = random()
        if p >= 0 and p <= 1 and math.exp(action_init-action_fin) > metropolis:
            # 受理
            n_accept += 1
        else:
            # 棄却
            p = backup_p
        
        print("{} {}".format(p, n_accept/i))


if __name__ == "__main__":
    main()