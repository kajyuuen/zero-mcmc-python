import math
from random import random

def main():
    n_iter = 1000000
    sum_y = 0

    for i in range(1, n_iter+1):

        x = random()
        y = math.sqrt(1-x**2)

        sum_y += y

        print("{} {}".format(i, sum_y/i))

if __name__ == "__main__":
    main()