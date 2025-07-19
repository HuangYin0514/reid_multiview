import glob
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io
from config_plot import *

OUTPUT_DIR = "./analysis/results/ablation"


def plot_parameter_lambda():
    # 设置数据 --------------------------------------------------------
    x_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    map_values = [56.32, 57.57, 58.16, 58.68, 58.59, 58.73, 58.76, 59.12, 59.00, 58.62, 58.69, 58.44, 58.45, 58.33]
    rank1_values = [67.24, 68.55, 69.46, 69.82, 69.86, 70.18, 70.63, 71.18, 70.27, 70.09, 70.50, 70.23, 70.23, 69.91]

    # 创建画布 --------------------------------------------------------
    LINE_NUM = 1
    ROW_NUM = 1
    FIGSIZE = (4 * ROW_NUM, 3 * LINE_NUM)
    fig, axis = plt.subplots(LINE_NUM, ROW_NUM, figsize=FIGSIZE, dpi=LATAX_DPI)
    subfig_axis = axis
    secondary_axis = subfig_axis.twinx()  #  创建双y轴

    # 绘制 mAP 曲线
    map_line = subfig_axis.plot(x_values, map_values, marker="o", markersize=MARKERSIZE, label="mAP(%)")
    subfig_axis.set_ylabel("mAP(%)")  #  设置y轴标签
    subfig_axis.set_xlabel(r"Parameter $\lambda$")

    offset = 0.5
    subfig_axis.set_ylim(min(map_values) - offset, max(map_values) + offset * 5)  # 设置 y 轴范围

    # 绘制 Rank-1 曲线
    rank1_line = secondary_axis.plot(x_values, rank1_values, color="#ff7f0e", marker="o", markersize=MARKERSIZE, label="Rank-1(%)")
    secondary_axis.set_ylabel("Rank-1(%)")
    offset = 0.5
    secondary_axis.set_ylim(min(rank1_values) - offset * 5, max(rank1_values) + offset)  # 设置 y 轴范围

    # 添加图例
    lines = (map_line[0], rank1_line[0])
    lables = [name.get_label() for name in lines]
    secondary_axis.legend(lines, lables)
    secondary_axis.legend(lines, lables).get_frame().set_alpha(1)  # 设置不透明度

    # 保存
    plt.tight_layout()
    # plt.show()
    output_file = os.path.join(OUTPUT_DIR, "parameter_lambda.png")
    plt.savefig(output_file, bbox_inches="tight")
    print(f"图像已保存到: {output_file}")


if __name__ == "__main__":
    # Path check
    OUTPUT_DIR = Path(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # parameter A
    plot_parameter_lambda()
