"""
Scripts to visualize layer importance
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab  as pylab

def load_plot_config():
    plt.style.use("seaborn-v0_8-paper")
    font_size=16
    legend_font_size = 12
    params = {'legend.fontsize': legend_font_size,
              'figure.figsize': (7, 4),
             'axes.labelsize': font_size,
             'axes.titlesize': font_size,
             'xtick.labelsize':font_size,
             'ytick.labelsize':font_size,
             'lines.linewidth': 2,
             'lines.markersize': 12,
             'font.weight': 500}
    pylab.rcParams.update(params)


def load_data():
    df = pd.read_csv("data/layer_importance.csv")
    res = {"keep 1/2": df.iloc[:, 0].tolist(), 
           "keep 1/4": df.iloc[:, 1].tolist(),
           "keep 1/8": df.iloc[:, 2].tolist()}
    return res


if __name__ == "__main__":
    load_plot_config()
    # key: label, value: list
    res = load_data()

    fig, ax = plt.subplots(1, 1)
    for label, data in res.items():
        ax.plot([i+1 for i in range(len(data))], data, "-o", label=label)

    ax.axhline(52.14, linestyle="--", label="BLIP-2 w/o pruning", color='gray')
    ax.set_ybound(lower=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("i-th cross-attention layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig("res/layer_importance.png")
    

