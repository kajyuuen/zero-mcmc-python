import math
from random import random

def main():
    n_iter = 100000

    sum_z = 0
    n_in = 0

    for i in range(1, n_iter+1):

        x, y = random(), random()

        if x**2 + y**2 < 1:
            n_in += 1
            z = math.sqrt(1-x**2-y**2)
            sum_z += z

        print("{} {}".format(i, sum_z/n_in*2*math.pi))

if __name__ == "__main__":
    main()