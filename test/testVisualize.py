import matplotlib.pyplot as plt
import numpy as np

def visualize_vrptw_clusters(coordinates, clusters, draw_routes=True):
    """
    Visualize kết quả phân cụm cho bài toán VRPTW
    - Màu sắc đậm, tương phản cao
    - Hiển thị số trực tiếp trên điểm
    """

    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    depot = coordinates[0]

    # Dùng colormap có độ tương phản cao
    cmap = plt.cm.get_cmap('tab10', len(clusters))  # tab10 ít màu hơn nhưng đậm hơn tab20
    colors = [cmap(i) for i in range(len(clusters))]

    # Vẽ depot
    ax.plot(depot[0], depot[1], 'r*', markersize=12, label='Depot', zorder=5)
    ax.annotate('D', (depot[0], depot[1]), fontsize=8, fontweight='bold',
                xytext=(2, 2), textcoords='offset points')

    # Vẽ từng cụm
    for idx, cluster in enumerate(clusters):
        color = colors[idx]
        cluster_coords = coordinates[cluster]
        xs, ys = cluster_coords[:, 0], cluster_coords[:, 1]

        # Điểm khách hàng — màu đậm, viền đen rõ ràng
        ax.scatter(xs, ys, c=[color], s=70, label=f'C{idx+1} ({len(cluster)})',
                   alpha=1.0, edgecolors='black', linewidths=1.2, zorder=3)
        
        # Đánh số điểm trực tiếp
        for i, point_idx in enumerate(cluster):
            ax.annotate(str(point_idx),
                        (coordinates[point_idx][0], coordinates[point_idx][1]),
                        fontsize=7, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', 
                                  facecolor='white', edgecolor=color, alpha=0.8))
        
        # Vẽ đường nối (route)
        if draw_routes and cluster:
            first_point = coordinates[cluster[0]]
            last_point = coordinates[cluster[-1]]

            # Depot -> điểm đầu
            ax.plot([depot[0], first_point[0]], [depot[1], first_point[1]],
                    c=color, linestyle='--', alpha=0.7, linewidth=1.3)
            # Các điểm trong cụm
            for i in range(len(cluster) - 1):
                start = coordinates[cluster[i]]
                end = coordinates[cluster[i + 1]]
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        c=color, alpha=0.9, linewidth=1.8)
            # Điểm cuối -> Depot
            ax.plot([last_point[0], depot[0]], [last_point[1], depot[1]],
                    c=color, linestyle='--', alpha=0.7, linewidth=1.3)

    # Tùy chỉnh hiển thị
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_title('VRPTW Clusters', fontsize=10, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=7, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    # ===== Thống kê =====
    print("\n" + "="*50)
    print("THỐNG KÊ PHÂN CỤM")
    print("="*50)
    print(f"Tổng số cụm (xe): {len(clusters)}")
    print(f"Tổng số khách hàng: {sum(len(c) for c in clusters)}")
    print("\nChi tiết từng cụm:")
    for idx, cluster in enumerate(clusters):
        print(f"  Cluster {idx+1}: {len(cluster)} điểm - Route: D -> {' -> '.join(map(str, cluster))} -> D")


# ===== Ví dụ =====
if __name__ == "__main__":
    coordinates = np.array([
        [40, 50],
        [45, 68], [45, 70], [42, 66],
        [42, 68], [42, 65], [40, 69],
        [38, 68], [38, 70], [35, 66],
        [35, 69], [25, 85], [22, 75],
        [22, 85], [20, 80], [20, 85],
        [18, 75], [15, 75], [15, 80],
        [30, 50], [30, 52], [28, 52],
        [28, 55], [25, 50], [25, 52],
    ])

    clusters = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21, 22, 23, 24]
    ]

    visualize_vrptw_clusters(coordinates, clusters)
