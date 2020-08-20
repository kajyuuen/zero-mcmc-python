import math
from random import random

def main():
    n_iter = 100
    step_size = 1/2
    
    x = 0
    n_accept = 0

    for i in range(1, n_iter+1):
        backup_x = x

        # S(x) = (x^2)/2
        action_init = (1/2)*(x**2)
        
        # dx ~ Uniform(-step_size, step_size)
        dx = random()
        dx = (dx-1/2)*step_size*2
        
        x += dx
        
        # S(x') = (x'^2)/2
        action_fin=(1/2)*(x**2) 
        
        # メトロポリステスト
        metropolis = random()
        if math.exp(action_init-action_fin) > metropolis:
            # 受理
            n_accept+= 1
        else:
            # 棄却
            x = backup_x

        print("{} {}".format(x, n_accept/i))

if __name__ == "__main__":
    main()