"""
Module vẽ biểu đồ so sánh kết quả các thuật toán VRPTW
======================================================
Hỗ trợ 2 phương thức vẽ biểu đồ:
  1. `draw_comparison_from_data(...)`: Nhận dữ liệu `data_excels` trực tiếp sau khi chạy xong ở main.py.
  2. `draw_comparison_from_excel(...)`: Đọc dữ liệu từ file Excel kết quả đã xuất để vẽ biểu đồ.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from ultility.utilities import plot_comparison_results


def draw_comparison_from_data(
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
    Vẽ biểu đồ so sánh từ mảng kết quả `data_excels` sau khi chạy main.py.
    """
    return plot_comparison_results(
        data_excels=data_excels,
        data_files=data_files,
        algorithms=algorithms,
        title_names=title_names,
        data_name=data_name,
        cluster_label=cluster_label,
        save_path=save_path,
        show=show,
    )


def draw_comparison_from_excel(
    excel_path: str,
    title_names: list = None,
    save_path: str = None,
    show: bool = True,
):
    """
    Đọc dữ liệu từ file Excel kết quả (được xuất từ main.py) và vẽ biểu đồ so sánh.

    Parameters
    ----------
    excel_path : str
        Đường dẫn đến file .xlsx xuất từ main.py.
    title_names : list of str (optional)
        Danh sách độ đo cần vẽ (mặc định: ['Route', 'Distance', 'Fitness', 'RunTime']).
    save_path : str (optional)
        Đường dẫn lưu file ảnh (PNG).
    show : bool
        Hiển thị biểu đồ (mặc định: True).
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Không tìm thấy file Excel: {excel_path}")

    if title_names is None:
        title_names = ['Route', 'Distance', 'Fitness', 'RunTime']

    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names

    if not sheet_names:
        raise ValueError("File Excel không có worksheet nào.")

    # Đọc cấu trúc từ sheet đầu tiên
    df_first = pd.read_excel(excel_path, sheet_name=sheet_names[0], header=None)

    # Dòng 1 (index 1) chứa tên các instance (bỏ qua cột đầu tiên 'Lần chạy')
    raw_instance_row = df_first.iloc[1, 1:].dropna().values
    instances = []
    for inst in raw_instance_row:
        inst_str = str(inst).strip()
        if inst_str and inst_str not in instances:
            instances.append(inst_str)

    n_metrics = len(title_names)
    algorithms_data = {}

    for sheet_name in sheet_names:
        df_sheet = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        
        # Tìm dòng trung bình "TB"
        tb_row_mask = df_sheet.iloc[:, 0].astype(str).str.strip().str.upper() == 'TB'
        if not tb_row_mask.any():
            # Nếu không có dòng TB, lấy dòng cuối cùng
            tb_row_values = df_sheet.iloc[-1, 1:].values
        else:
            tb_row_values = df_sheet[tb_row_mask].iloc[0, 1:].values

        # Phân rã dữ liệu: mỗi instance có n_metrics giá trị
        alg_metrics = {m: [] for m in title_names}
        for i in range(len(instances)):
            for m_idx, m_name in enumerate(title_names):
                col_idx = i * n_metrics + m_idx
                if col_idx < len(tb_row_values):
                    val = float(tb_row_values[col_idx])
                else:
                    val = np.nan
                alg_metrics[m_name].append(val)

        algorithms_data[sheet_name] = alg_metrics

    # Tiến hành vẽ biểu đồ
    nrows, ncols = (2, 2) if n_metrics > 2 else (n_metrics, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 11))
    axes_flat = np.atleast_1d(axes).flatten()

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x', '+']
    cmap = plt.cm.get_cmap('tab10', max(len(algorithms_data), 1))

    for m_idx, metric_name in enumerate(title_names):
        ax = axes_flat[m_idx]
        for a_idx, (alg_name, metrics_dict) in enumerate(algorithms_data.items()):
            vals = metrics_dict[metric_name]
            marker = markers[a_idx % len(markers)]
            color = cmap(a_idx)
            ax.plot(
                instances, vals,
                label=alg_name,
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

        if len(instances) > 8:
            ax.tick_params(axis='x', rotation=45)

    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    fig.suptitle(f"So sánh kết quả từ file: {base_name}", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    # Dữ liệu mẫu kiểm thử
    data_R = {
        "Route": [
            "R101", "R102", "R103", "R104", "R105", "R106", "R107", "R108", "R109", "R110",
            "R111", "R112", "R201", "R202", "R203", "R204", "R205", "R206", "R207", "R208",
            "R209", "R210", "R211"
        ],
        "Kmeans+GA_Distance": [
            1596.05, 1554.66, 1401.75, 1193.30, 1571.18, 1510.12, 1292.15, 1159.76, 1494.83,
            1390.72, 1280.47, 1193.91, 1547.97, 1556.80, 1556.12, 1476.48, 1554.37, 1543.29,
            1578.05, 1484.47, 1540.82, 1597.71, 1493.82
        ],
        "Modify_Kmeans+GA_Distance": [
            1588.60, 1552.66, 1406.62, 1193.72, 1575.43, 1509.16, 1296.03, 1162.97, 1494.15,
            1386.00, 1289.93, 1193.88, 1854.90, 1802.40, 1859.06, 1485.08, 1751.21, 1731.47,
            1706.47, 1276.79, 1624.55, 1818.26, 1388.21
        ],
        "Kmeans+GA_Fitness": [
            2521.57, 1988.17, 1621.29, 1293.75, 1977.07, 1649.26, 1422.45, 1213.99, 1641.91,
            1461.90, 1382.32, 1260.44, 6963.35, 6729.41, 6377.44, 4817.74, 6228.66, 5968.33,
            5604.75, 4123.98, 5273.74, 6230.59, 4130.70
        ],
        "Modify_Kmeans+GA_Fitness": [
            2514.21, 1978.98, 1619.53, 1293.10, 1973.59, 1649.96, 1429.20, 1213.35, 1639.54,
            1458.56, 1385.48, 1262.43, 2713.68, 2577.90, 2406.62, 1667.44, 2158.46, 2109.17,
            2079.47, 1366.21, 1863.55, 2194.88, 1435.15
        ],
        "Kmeans+GA_RunTime": [
            211.43, 215.53, 214.25, 210.82, 214.06, 195.62, 176.87, 137.81, 136.97, 129.87,
            127.24, 126.93, 175.57, 184.08, 179.84, 181.90, 179.76, 179.15, 179.18, 155.84,
            151.45, 155.54, 153.89
        ],
        "Modify_Kmeans+GA_RunTime": [
            157.21, 158.58, 212.95, 210.70, 212.36, 209.27, 213.99, 195.75, 178.14, 145.07,
            138.01, 129.97, 613.99, 631.11, 550.13, 525.57, 724.72, 846.11, 598.05, 500.44,
            499.32, 488.36, 494.99
        ],
        "Kmeans+GA_Route": [
            16.2, 14.9, 13.3, 13, 15, 14, 13, 13, 15, 14, 13, 13, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10
        ],
        "Modify_Kmeans+GA_Route": [
            16, 15, 13.4, 13, 15, 14, 13, 13, 15, 14, 13, 13, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4
        ]
    }
    df = pd.DataFrame(data_R)
    print("DataFrame mẫu:")
    print(df.head())
