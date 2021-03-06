import const

import pandas as pd
from matplotlib import pyplot as plt


def plot_history_rolling_mean(hist, N=const.rolling_mean_N, fp=const.file_path_img_score):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # prepare data
    x = pd.Series(hist)
    y = x.rolling(window=N).mean().iloc[N - 1:]

    # plot
    plt.plot(hist, c='darkorchid', marker='.', markevery=[-1])
    plt.plot(y, c='blue', marker='.', markevery=[-1])

    # plot line to signify the aimed high score
    x1, x2 = 0, len(hist)
    y1, y2 = const.high_score, const.high_score
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='--', linewidth=1)

    # annotate last_history and last_rolling point
    last_points = [(len(hist) - 1, hist[-1])]
    if not y.empty:
        last_points.append((len(hist) - 1, y.iloc[-1]))

    for (i, j) in last_points:
        ax.annotate('{:.2f}'.format(j), xy=(i, j), xytext=(i + 0.1, j))

    plt.xlabel('Episodes')
    plt.ylabel('Score (Sum of Rewards)')
    plt.title('Online Performance')
    plt.legend(['score', 'rolling_score (N={})'.format(N)], loc='best')
    plt.savefig(fp)
    plt.close()
