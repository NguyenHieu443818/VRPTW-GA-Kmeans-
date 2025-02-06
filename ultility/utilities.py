# demo các hàm trong phương thức
import numpy as np
import array
from numpy import ndarray

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

# Ma trận độ thuộc ra nhãn (giải mờ)
def extract_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1)

# # Chia các điểm vào các cụm
# def extract_clusters(data: np.ndarray, labels: np.ndarray, n_cluster: int = 0) -> list:
#     if n_cluster == 0:
#         n_cluster = np.unique(labels)
#     return [data[labels == i] for i in range(n_cluster)]
# Chia các điểm vào các cụm
def extract_clusters(labels: np.ndarray, n_cluster: int = 0) -> list:
    if n_cluster == 0:
        n_cluster = np.unique(labels)
    return [np.argwhere([labels == i]).T[1,] + 1 for i in range(n_cluster)]

# Làm tròn số
def round_float(number: float, n: int = 2) -> float:
    if n == 0:
        return int(number)
    return round(number, n)
# Biểu diễn các điểm lên


def visualize_clusters(data, labels, centers):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.scatter(data[0, 0], data[0, 1], c='black',
                marker='o', s=200, label='Starter')
    # Xóa hàng đầu tiên
    data = np.delete(data, 0, 0)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1], label=f'Cluster {label}')
    # Convert centers to a NumPy array for proper indexing
    centers_array = np.array(centers)
    print(np.shape(centers_array))

    plt.scatter(centers_array[:, 0], centers_array[:, 1],
                c='red', marker='x', s=200, label='Centers')
    plt.legend()
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def write_excel_file(data_excels, data_files, data_name, run_time, algorithms, title_names, fileio):
    from xlsxwriter import Workbook
    import numpy as np

    workbook = Workbook(fileio)
    for idx_a, algorithm in enumerate(algorithms):
        worksheet = workbook.add_worksheet(data_name+algorithm+str(idx_a))
        titformat = workbook.add_format(
            {'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 14})
        char_data_end = (len(data_files)*len(title_names) // 26) * \
            "A" + chr(ord('A')+len(data_files)*len(title_names) % 26)
        worksheet.merge_range(
            f'A1:{char_data_end}1', f'Kết quả chạy thử bộ dữ liệu bằng {algorithm}', titformat)
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
        
        data_mean = np.mean(data,axis=0,keepdims=True) # Tính giá trị trung bình cho các lần tính
        data = np.append(data,data_mean,axis=0)

        for row, dr in enumerate(data):
            dat = [row + 1] # Tính cho hàng giá trị trung bình
            dat = dat + list(dr)
            for i, item in enumerate(dat):
                worksheet.write(row + 3, i, item, colformat)
        worksheet.write(row+3, 0, "TB", colformat)
    workbook.close()
