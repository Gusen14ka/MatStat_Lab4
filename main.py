from distributions import *
from plotting import plot
import matplotlib.pyplot as plt

def main():
    sample_size = [20, 60, 100]

    distribustions = {
        #"Normal": generate_normal,
        "Cauchy": generate_cauchy,
        #"Laplace": generate_laplace,
        "Poisson": generate_poisson,
        #"Uniform": generate_uniform
    }


    for n in sample_size:
        for name, gen in distribustions.items():
            plot(gen(n), name)

    plt.show()
if __name__ == "__main__":
    main()