import numpy as np
import matplotlib.pyplot as plt
from C2C.UMAP_visual.dataloader_numpy import *


if __name__ == "__main__":
# Load embedding from the file
    embedding = np.load('F:/save_ficture/draw_umap/embedding_imagnet_samplingpoints.npy')
    data, labels = load_data()
    # 4. 可视化
    colors = ['#dd1c77', '#1a9641', '#0571b0', '#ff7f00']
    labels_unique = np.unique(labels)
    plt.figure(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(right=0.9)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[colors[int(label)] for label in labels], s=14)  # 绘制散点图，按照聚类结果着色
    # Add legend
    labels = ['LMS', 'SS', 'US', 'LS']
    for i, label in enumerate(np.unique(labels)):
        plt.scatter([], [], c=colors[i], label=labels[i])  # Create empty scatter plot for legend
    #添加图例
    plt.legend(frameon=True, loc='lower left', fontsize=11)  # Show legend

    #plt.title('UMAP Visualization')
    plt.xlabel('Dimension 1', fontsize=13)
    plt.ylabel('Dimension 2', fontsize=13)
    #plt.colorbar()
    # 隐藏轴的刻度
    plt.xticks([])
    plt.yticks([])

    plt.savefig('F:/save_ficture/draw_umap/imagnet_points_umap.png')
    plt.show()
