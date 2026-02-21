import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from kde import *

def plot(sample, name): 
    n = np.size(sample)
    
    # set plotting range depending on distribution
    if name == "Poisson":
        low, high = 6, 14
    else:
        low, high = -4, 4

    if name == "Cauchy":
        h = robast_silverman_bandwidth(sample)
    else:
        h = silverman_bandwidth(sample)

    

    plt.figure()
    plt.grid()
    
    if name == "Poisson":
        values = np.arange(low, high + 1)
        counts = np.array([np.sum(sample == k) for k in values])
        probs = counts / n

        plt.bar(values, probs, alpha=0.67, edgecolor="black", 
                color="skyblue", label="Гистограмма")

        teor_probs = st.poisson.pmf(values, 5)
        plt.vlines(values, 0, teor_probs, color="red", linewidth=3, label="Теория")
        plt.xlim(low, high)

    #elif name == ca

    else:
        x = np.linspace(low, high, 1000)
        plt.hist(sample, bins="fd", range=(low, high), density=True,
            alpha=0.67, edgecolor="black", color="skyblue", label="Генерация")
        #Теор плотность
        pdf_functions = {
            "Normal": lambda x: st.norm.pdf(x, 0, 1),
            "Laplace": lambda x: st.laplace.pdf(x, 0, np.sqrt(1/2)),
            "Uniform": lambda x: st.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)),
            "Cauchy": lambda x: st.cauchy.pdf(x, 0, 1)
        }

        y = pdf_functions[name](x)

        plt.plot(x, y, color="red", label="Теория")

        kde_vals = gaussian_kde(x, sample, h)
        plt.plot(x, kde_vals, label="Ядерная оценка", color="orange")
        plt.xlim(low, high)

    
    plt.title(f"Распределение {name}, n={n}")
    plt.xlabel("Значение случайной величины")
    plt.ylabel("Плотность вероятности")
    plt.legend()

    #ЭФР и функция
    if name == "Poisson":
        x = np.linspace(low, high, 1000)
    else:
        x = np.linspace(low, high, 1000)
    plt.figure()
    plt.grid()
    sample_sorted = np.sort(sample)
    y_all = np.arange(1, n + 1) / n
    mask = (sample_sorted >= low) & (sample_sorted <= high)
    x_ecdf = np.concatenate(([low], sample_sorted[mask], [high]))
    y_ecdf = np.concatenate(([np.sum(sample_sorted <= low) / n], y_all[mask], [np.sum(sample_sorted <= high) / n]))
    plt.step(x_ecdf, y_ecdf, color="blue", where="post", label="ЭФР")
    plt.xlim(low, high)

    #Теор функция
    pdf_functions = {
        "Normal": lambda x: st.norm.cdf(x, 0, 1),
        "Laplace": lambda x: st.laplace.cdf(x, 0, np.sqrt(1/2)),
        "Uniform": lambda x: st.uniform.cdf(x, -np.sqrt(3), 2*np.sqrt(3)),
        "Cauchy": lambda x: st.cauchy.cdf(x, 0, 1),
        "Poisson": lambda x: st.poisson.cdf(x, 5)
    }

    y = pdf_functions[name](x)

    plt.plot(x, y, color="red", label="Теория")

    plt.title(f"Распределение {name}, n={n}")
    plt.xlabel("Значение случайной величины")
    plt.ylabel("F(x)")
    plt.legend()
    


        


