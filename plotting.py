import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from kde import *

def plot(sample, name):
    n = np.size(sample)

    if name == "Poisson":
        low, high = 6, 14
    else:
        low, high = -4, 4

    span = high - low
    margin = 0.05 * span
    lim_low, lim_high = low - margin, high + margin

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

        teor_probs = st.poisson.pmf(values, 10)
        plt.vlines(values, 0, teor_probs, color="red", linewidth=3, label="Теория")
        plt.xlim(lim_low, lim_high)

    else:
        x = np.linspace(low, high, 1000)
        plt.hist(sample, bins="fd", range=(low, high), density=True,
                 alpha=0.67, edgecolor="black", color="skyblue", label="Генерация")
        # теоретическая плотность
        pdf_functions = {
            "Normal": lambda x: st.norm.pdf(x, 0, 1),
            "Laplace": lambda x: st.laplace.pdf(x, 0, np.sqrt(1/2)),
            "Uniform": lambda x: st.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)),
            "Cauchy": lambda x: st.cauchy.pdf(x, 0, 1)
        }

        y = pdf_functions[name](x)
        plt.plot(x, y, color="red", label="Теория")

        kde_full = gaussian_kde(x, sample, h, m=n)
        plt.plot(x, kde_full, label=f"Ядерная оценка (m={n})", color="purple")
        if n > 1:
            kde_half = gaussian_kde(x, sample, h, m=n // 2)
            plt.plot(x, kde_half, label=f"Ядерная оценка (m={n//2})",
                     color="orange", alpha=0.6, linestyle="--")
        if n > 3:
            kde_quarter = gaussian_kde(x, sample, h, m=n // 4)
            plt.plot(x, kde_quarter, label=f"Ядерная оценка (m={n//4})",
                     color="green", alpha=0.6, linestyle=":")

        plt.xlim(lim_low, lim_high)

    plt.title(f"Распределение {name}, n={n}")
    plt.xlabel("Значение случайной величины")
    plt.ylabel("Плотность вероятности")
    plt.legend()

    # ЭФР и сравнение с теорией (ограничено строго [low,high])
    plt.figure()
    plt.grid()
    sample_sorted = np.sort(sample)
    y_all = np.arange(1, n + 1) / n

    # Добавляем точки на границах, чтобы ЭФР визуально начиналась в 0 и заканчивалась в 1
    ecdf_left = min(low, float(sample_sorted[0]))
    ecdf_right = max(high, float(sample_sorted[-1]))
    x_ecdf = np.concatenate(([ecdf_left], sample_sorted, [ecdf_right]))
    y_ecdf = np.concatenate(([0], y_all, [1]))
    plt.step(x_ecdf, y_ecdf, color="blue", where="post", label="ЭФР")

    if name == "Poisson":
        values = np.arange(low, high + 1)
        counts = np.array([np.sum(sample == k) for k in values])
        probs = counts / n
        cum_probs = np.cumsum(probs)
        x_hist = np.concatenate(([low], values, [high]))
        y_hist = np.concatenate(([0], cum_probs, [1]))
        plt.step(x_hist, y_hist, where="post", color="green", label="ЭФР (гист)")
    else:
        counts, bin_edges = np.histogram(sample, bins="fd", range=(low, high))
        probs = counts / n
        cum_probs = np.cumsum(probs)
        x_hist = np.concatenate(([low], bin_edges[1:], [high]))
        y_hist = np.concatenate(([0], cum_probs, [1]))
        plt.step(x_hist, y_hist, where="post", color="green", label="ЭФР (гист)")

    #plt.xlim(low, high)

    # теоретическая функция распределения
    pdf_functions = {
        "Normal": lambda x: st.norm.cdf(x, 0, 1),
        "Laplace": lambda x: st.laplace.cdf(x, 0, np.sqrt(1/2)),
        "Uniform": lambda x: st.uniform.cdf(x, -np.sqrt(3), 2*np.sqrt(3)),
        "Cauchy": lambda x: st.cauchy.cdf(x, 0, 1),
        "Poisson": lambda x: st.poisson.cdf(x, 10)
    }

    x = np.linspace(ecdf_left, ecdf_right, 1000)
    y = pdf_functions[name](x)
    plt.plot(x, y, color="red", label="Теория")

    plt.title(f"Распределение {name}, n={n}")
    plt.xlabel("Значение случайной величины")
    plt.ylabel("F(x)")
    plt.legend()
    


        


