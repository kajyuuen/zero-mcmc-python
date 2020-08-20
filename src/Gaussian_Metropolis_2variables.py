import math
from random import random

def main():
    n_iter = 10000
    step_size_x = 1/2
    step_size_y = 1/2
    
    x = 0
    y = 0
    n_accept = 0

    for i in range(1, n_iter+1):
        backup_x = x
        backup_y = y

        action_init = (1/2)*(x**2+y**2+x*y)
        
        # dx ~ Uniform(-step_size_x, step_size_x)
        # dy ~ Uniform(-step_size_y, step_size_y)
        dx = random()
        dy = random()
        dx = (dx-1/2)*step_size_x*2
        dy = (dy-1/2)*step_size_x*2
        
        x += dx
        y += dy
        
        action_fin = (1/2)*(x**2+y**2+x*y)
        
        # メトロポリステスト
        metropolis = random()
        if math.exp(action_init-action_fin) > metropolis:
            # 受理
            n_accept+= 1
        else:
            # 棄却
            x = backup_x
            y = backup_y

        if i % 10 == 0:
            print("{} {} {}".format(x, y, n_accept/i))

if __name__ == "__main__":
    main()