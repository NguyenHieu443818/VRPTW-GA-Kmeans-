# demo các hàm trong phương thức
import numpy as np
import array
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Kiểm tra kiểu dữ liệu
def check_data_type(obj):
    if isinstance(obj, list):
        return "List"
    elif isinstance(obj, tuple):
        return "Tuple"
    elif isinstance(obj, dict):
        return "Dictionary"
    elif isinstance(obj, array.array):
        return "Array"
    elif isinstance(obj, np.ndarray):
        return "ndarray"
    else:
        return "Không phải kiểu dữ liệu được kiểm tra"
# Tính khoảng cách từ 1 điểm đến tất cả các điểm còn lại bằng euclidean
def distance_cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    # return distance_euclidean(X,Y) if metric=='euclidean' else distance_chebyshev(X,Y)
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)

# Trích xuất các cụm từ nhãn
def extract_clusters(labels: np.ndarray, n_cluster: int = 0) -> list:
    if n_cluster == 0:
        n_cluster = np.unique(labels)
    return [np.argwhere([labels == i]).T[1,] + 1 for i in range(n_cluster)]

# Làm tròn số
def round_float(number: float, n: int = 2) -> float:
    if n == 0:
        return int(number)
    return round(number, n)


def create_graph(coords):
    """
    Tạo ma trận khoảng cách từ danh sách tọa độ.
    :param coords: Danh sách tọa độ (danh sách các tuple dạng (x, y))
    :return: Ma trận khoảng cách
    """
    return distance_cdist(coords, coords)

def visualize_vrptw_clusters(coordinates, clusters, draw_routes=True):
    """
    Visualize kết quả phân cụm cho bài toán VRPTW
    - Màu sắc đậm, tương phản cao, đủ cho nhiều cụm
    - Hiển thị số trực tiếp trên điểm
    """

    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    depot = coordinates[0]

    # ✅ Dải màu tương phản cao, đủ cho nhiều cụm (>= 50)
    cmap = plt.cm.get_cmap('nipy_spectral', len(clusters))
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


def visualize_fuzzy_clusters(
    cord_data: np.ndarray,
    U: np.ndarray,
    labels: np.ndarray = None,
    medoids: list = None,
    kappa: float = 2.0,
    n_iter: int = 0,
    title_suffix: str = '',
    show: bool = True,
):
    """
    Hiển thị kết quả phân cụm mờ (Fuzzy Clustering / Fuzzy c-Medoids) dưới dạng scatter plot tương tác.

    - Mỗi điểm được tô màu theo cụm cứng (hard assignment).
    - Medoid (nếu có) hiển thị bằng biểu tượng hình thoi ◆.
    - **Di chuột** vào một điểm → tooltip hiện top 4 cụm có
      độ thuộc về mờ (μ) cao nhất của điểm đó.

    Parameters
    ----------
    cord_data    : np.ndarray, shape (n+1, 2)
        Cột 0 = X, cột 1 = Y. Hàng 0 là depot, từ hàng 1 là khách hàng.
    U            : np.ndarray, shape (n, q)
        Ma trận độ thuộc về mờ.
    labels       : np.ndarray, shape (n,) (optional)
        Nhãn cụm cứng (0-indexed). Nếu None sẽ lấy argmax(U, axis=1).
    medoids      : list of int (optional)
        Danh sách chỉ số 1-indexed của các medoid/tâm cụm.
    kappa        : float
        Hệ số mờ.
    n_iter       : int
        Số vòng lặp đã chạy.
    title_suffix : str
        Chuỗi bổ sung vào tiêu đề (ví dụ: tên file dữ liệu).
    show         : bool
        Gọi plt.show() nếu True.

    Returns
    -------
    (fig, ax) – đối tượng matplotlib Figure và Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not isinstance(cord_data, np.ndarray):
        cord_data = np.array(cord_data)

    n = U.shape[0]  # số khách hàng
    q = U.shape[1]  # số cụm
    if labels is None:
        labels = np.argmax(U, axis=1)

    coords = cord_data[1:]  # (n, 2) — bỏ depot
    depot = cord_data[0]   # (2,)
    n_show = min(4, q)     # top k cụm hiển thị trong tooltip

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(right=0.78)

        cmap = plt.cm.get_cmap('tab20', q)

        # ── Khách hàng — tô màu theo cụm cứng ────────────────────
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap='tab20', vmin=0, vmax=max(q - 1, 1),
            s=65, zorder=3,
            edgecolors='white', linewidths=0.4,
        )

        # Số thứ tự khách hàng (nhỏ, trong suốt)
        for i in range(n):
            ax.annotate(
                str(i + 1), coords[i],
                fontsize=5, color='#cccccc', alpha=0.65, zorder=4,
                xytext=(2, 2), textcoords='offset points',
            )

        # ── Depot ─────────────────────────────────────────────────
        ax.scatter(depot[0], depot[1],
                   c='#ff4757', marker='*', s=280, zorder=6,
                   edgecolors='white', linewidths=0.8)
        ax.annotate('Depot', depot, fontsize=7, color='#ff4757',
                    xytext=(4, 4), textcoords='offset points', fontweight='bold')

        # ── Medoids (nếu có) ──────────────────────────────────────
        if medoids:
            for p in range(len(medoids)):
                mc = cord_data[medoids[p]]
                ax.scatter(mc[0], mc[1],
                           c=[cmap(p)], marker='D', s=130, zorder=5,
                           edgecolors='white', linewidths=1.5)

        # ── Annotation / Tooltip ──────────────────────────────────
        annot = ax.annotate(
            "", xy=(0, 0),
            xytext=(22, 22), textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.55",
                fc="#16213e", ec="#7c83ff", alpha=0.93, linewidth=1.4,
            ),
            fontsize=8.5, color='white',
            fontfamily='monospace',
            arrowprops=dict(
                arrowstyle="-|>",
                color="#7c83ff", lw=1.3,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=10,
        )
        annot.set_visible(False)

        def _update_annot(idx: int):
            cust_id = idx + 1
            u_row = U[idx]
            hard = int(labels[idx])
            top_idx = np.argsort(u_row)[::-1][:n_show]

            lines = [
                f"  KH #{cust_id:>3d}  ·  Cụm cứng: {hard + 1}",
                f"  {'─' * 28}",
                f"  Top {n_show} độ thuộc về mờ (μ):",
            ]
            for ci in top_idx:
                mu = u_row[ci]
                filled = round(mu * 12)
                bar = '▓' * filled + '░' * (12 - filled)
                marker = ' ◀' if ci == hard else '  '
                lines.append(f"  Cụm {ci+1:>2d}: {mu:.4f}  {bar}{marker}")

            annot.xy = coords[idx]
            annot.set_text("\n".join(lines))

        def _on_hover(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return
            cont, det = sc.contains(event)
            if cont:
                _update_annot(det["ind"][0])
                annot.set_visible(True)
            else:
                annot.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_hover)

        # ── Legend ────────────────────────────────────────────────
        legend_handles = [
            plt.scatter([], [], c='#ff4757', marker='*', s=100, label='Depot'),
        ]
        if medoids:
            legend_handles.append(
                plt.scatter([], [], c='white', marker='D', s=55,
                            edgecolors='gray', linewidths=1, label='Medoid')
            )
            legend_handles += [
                mpatches.Patch(
                    color=cmap(p),
                    label=f"Cụm {p+1}  (medoid #{medoids[p]})",
                )
                for p in range(q)
            ]
        else:
            legend_handles += [
                mpatches.Patch(color=cmap(p), label=f"Cụm {p+1}")
                for p in range(q)
            ]

        ax.legend(
            handles=legend_handles,
            loc='upper left', bbox_to_anchor=(1.01, 1),
            fontsize=7, framealpha=0.2,
            facecolor='#16213e', edgecolor='#7c83ff',
            labelcolor='white',
        )

        # ── Tiêu đề & nhãn trục ───────────────────────────────────
        subtitle = f"  ·  {title_suffix}" if title_suffix else ""
        iter_str = f"  ·  {n_iter} vòng lặp" if n_iter > 0 else ""
        ax.set_title(
            f"Fuzzy Clustering  ·  q={q} cụm  ·  κ={kappa}"
            f"  ·  {n} KH{iter_str}{subtitle}",
            fontsize=10, fontweight='bold', pad=12,
        )
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.12, linestyle=':', color='#aaaaaa')

        if show:
            plt.show()

    return fig, ax

    """
    Vẽ các tuyến đường trong bài toán VRPTW với mỗi tuyến (xe) có màu khác nhau.

    Parameters:
        coordinates (np.ndarray): Mảng numpy với shape (n, 2) chứa tọa độ [x, y] của depot và khách hàng.
        routes (list of list of int): Danh sách các tuyến, mỗi tuyến là list các chỉ số (int).
        depot_index (int): Chỉ số depot trong mảng tọa độ.
        save_path (str): Nếu muốn lưu ảnh, truyền vào đường dẫn file.
    """
    # Kiểm tra coordinates hợp lệ
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("'coordinates' phải là numpy array.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("'coordinates' phải có shape (n, 2) — mỗi hàng là [x, y].")

    n_points = coordinates.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    ax.scatter(x, y, c='black', label='Customer', zorder=3)
    ax.scatter(x[depot_index], y[depot_index], c='red', s=150, marker='*', label='Depot', zorder=4)

    # for i, (xi, yi) in enumerate(zip(x, y)):
    #     ax.annotate(str(i), (xi + 0.5, yi + 0.5), fontsize=8)

    color_map = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        if not isinstance(route, list):
            raise TypeError(f"Route {i+1} phải là list các chỉ số (int), nhưng nhận được: {type(route)}")

        for idx in route:
            if not isinstance(idx, int):
                raise TypeError(f"Route {i+1} chứa phần tử không phải int: {idx}")
            if not (0 <= idx < n_points):
                raise IndexError(f"Route {i+1} chứa chỉ số không hợp lệ: {idx}")

        route_coords = coordinates[route]
        ax.plot(route_coords[:, 0], route_coords[:, 1],
                label=f'Vehicle {i+1}', linewidth=0, color=color_map(i), marker='o')

    ax.set_title('VRPTW - Vehicle Routes Visualization')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def write_excel_file(data_excels, data_files, data_name, run_time, algorithms, title_names, fileio,
                     cluster_label: str = ''):
    from xlsxwriter import Workbook
    import numpy as np

    workbook = Workbook(fileio)
    for idx_a, algorithm in enumerate(algorithms):
        # Tên worksheet = data_name + algorithm + cluster_label + idx (tối đa 31 ký tự Excel)
        ws_name = (data_name + algorithm + cluster_label + str(idx_a))[:31]
        worksheet = workbook.add_worksheet(ws_name)
        titformat = workbook.add_format(
            {'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 14})
        char_data_end = (len(data_files)*len(title_names) // 26) * \
            "A" + chr(ord('A')+len(data_files)*len(title_names) % 26)
        alg_label = f"{algorithm}  [{cluster_label}]" if cluster_label else algorithm
        worksheet.merge_range(
            f'A1:{char_data_end}1', f'Kết quả chạy thử bộ dữ liệu bằng {alg_label}', titformat)
        hedformat = workbook.add_format(
            {'bold': 1, 'border': 1, 'align': 'center'})
        # In các bộ dữ liệu
        for idx_d, dat in enumerate(data_files):
            # Tính số lượng chữ cái A được lặp và chữ cái cuối cùng của chuỗi trong excel
            char_start = ((idx_d*len(title_names)+1) // 26)*"A" + \
                chr(ord('A')+((idx_d*len(title_names)+1) % 26))
            char_end = ((idx_d+1)*len(title_names) // 26)*"A" + \
                chr(ord('A')+((idx_d+1)*len(title_names) % 26))
            worksheet.merge_range(
                f'{char_start}2:{char_end}2', dat[:-4], titformat)

        # Tạo một list titles chứa các tiêu đề cho các cột
        titles = ['Lần chạy'] + (title_names*len(data_files))

        # Duyệt qua danh sách titles và ghi từng tiêu đề vào hàng thứ ba (chỉ số 2) của worksheet, áp dụng hedformat
        for idx_t, title in enumerate(titles):
            worksheet.write(2, idx_t, title, hedformat)

        # Tạo một đối tượng định dạng colformat cho viền của các ô
        colformat = workbook.add_format({'border': 1})

        # Định dạng lại dữ liệu đầu vào
        data = np.array(data_excels[idx_a])
        # số bộ dữ liệu x số lần chạy  x số thuộc tính
        data = np.reshape(
            data, (len(data_files), run_time, len(title_names)))

        # số lần chạy + 1 x (số bộ dữ liệu x số thuộc tính)
        data = data.transpose(1, 0, 2).reshape(
            run_time, len(data_files)*len(title_names))

        # Tính giá trị trung bình cho các lần tính
        data_mean = np.mean(data, axis=0, keepdims=True)
        data = np.append(data, data_mean, axis=0)

        for row, dr in enumerate(data):
            dat = [row + 1]  # Tính cho hàng giá trị trung bình
            dat = dat + list(dr)
            for i, item in enumerate(dat):
                worksheet.write(row + 3, i, item, colformat)
        worksheet.write(row+3, 0, "TB", colformat)
    workbook.close()


def plot_comparison_results(
    data_excels: np.ndarray,
    data_files: list,
    algorithms: list,
    title_names: list = None,
    data_name: str = "",
    cluster_label: str = "",
    save_path: str = None,
    show: bool = True,
):
    """
    Vẽ biểu đồ so sánh kết quả các thuật toán (tương tự Drawdata.py)
    sau khi chạy xong các thuật toán ở main.py.

    Biểu đồ gồm 4 subplot tương ứng với các độ đo:
      - Route (Số lượng xe)
      - Distance (Tổng quãng đường)
      - Fitness (Giá trị hàm thích nghi)
      - RunTime (Thời gian chạy - giây)

    Parameters
    ----------
    data_excels : np.ndarray
        Mảng dữ liệu 4D: (n_algorithms, n_files, n_runs, n_metrics)
        hoặc 3D: (n_algorithms, n_files, n_metrics).
    data_files : list of str
        Danh sách tên các file dữ liệu (ví dụ: ['R101.txt', 'R102.txt', ...]).
    algorithms : list of str
        Danh sách tên các thuật toán (ví dụ: ['GA_VNS', 'DiscretePSO']).
    title_names : list of str (optional)
        Danh sách tên các thuộc tính đo lường (mặc định: ['Route', 'Distance', 'Fitness', 'RunTime']).
    data_name : str (optional)
        Tên bộ dữ liệu (ví dụ: 'R1').
    cluster_label : str (optional)
        Nhãn thuật toán phân cụm (ví dụ: 'FCM' hoặc 'KMeans').
    save_path : str (optional)
        Đường dẫn lưu file ảnh (PNG).
    show : bool
        Hiển thị cửa sổ biểu đồ (mặc định: True).

    Returns
    -------
    (fig, axes) : matplotlib Figure và Axes
    """
    import matplotlib.pyplot as plt
    import os

    if title_names is None:
        title_names = ['Route', 'Distance', 'Fitness', 'RunTime']

    # Xử lý tên các instance trên trục X (bỏ phần đuôi .txt hoặc .csv)
    instance_names = [os.path.splitext(os.path.basename(f))[0] for f in data_files]

    # Tính giá trị trung bình qua các lần chạy (nếu là mảng 4D)
    data = np.asarray(data_excels)
    if data.ndim == 4:
        # (n_algorithms, n_files, n_runs, n_metrics) -> (n_algorithms, n_files, n_metrics)
        data_mean = np.mean(data, axis=2)
    elif data.ndim == 3:
        data_mean = data
    else:
        raise ValueError(f"data_excels phải có 3 hoặc 4 chiều, nhận được {data.ndim} chiều.")

    n_metrics = min(len(title_names), data_mean.shape[2])
    n_algorithms = len(algorithms)

    # Marker & màu sắc cho các thuật toán
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x', '+']
    cmap = plt.cm.get_cmap('tab10', max(n_algorithms, 1))

    # Cấu hình layout subplot 2x2 (hoặc 1xN nếu ít metric)
    if n_metrics <= 2:
        nrows, ncols = n_metrics, 1
        figsize = (12, 5 * n_metrics)
    else:
        nrows, ncols = 2, 2
        figsize = (15, 11)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    clustering_str = f" - Phân cụm: {cluster_label}" if cluster_label else ""
    header_title = f"So sánh kết quả thuật toán trên bộ dữ liệu {data_name}{clustering_str}"

    for m_idx in range(n_metrics):
        ax = axes_flat[m_idx]
        metric_name = title_names[m_idx]

        for a_idx, alg in enumerate(algorithms):
            vals = data_mean[a_idx, :, m_idx]
            marker = markers[a_idx % len(markers)]
            color = cmap(a_idx)
            label = f"{alg} ({cluster_label})" if cluster_label else alg

            ax.plot(
                instance_names, vals,
                label=label,
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=6,
                alpha=0.9,
            )

        ax.set_title(f"So sánh {metric_name}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Bộ dữ liệu (Instance)", fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='best')

        # Xoay nhãn trục x nếu có nhiều điểm
        if len(instance_names) > 8:
            ax.tick_params(axis='x', rotation=45)

    # Ẩn các axes thừa nếu có
    for extra_idx in range(n_metrics, len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])

    fig.suptitle(header_title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig, axes

