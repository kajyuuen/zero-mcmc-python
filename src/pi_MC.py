from random import random

def main():
    n_iter = 1000000
    n_in = 0

    for i in range(1, n_iter+1):

        x, y = random(), random()

        if x**2 + y**2 < 1:
            n_in += 1

        print("{} {}".format(i, n_in/i))

if __name__ == "__main__":
    main()