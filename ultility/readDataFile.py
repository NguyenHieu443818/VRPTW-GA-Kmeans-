import numpy as np
import os

C_ID = 0
C_X = 1
C_Y = 2
C_DEMAND = 3
C_READY_TIME = 4
C_DUE_TIME = 5
C_SERVICE_TIME = 6


def load_txt_dataset(url=None, name_of_id=None):
    """
    Tải dữ liệu từ file dataset Solomon, tối ưu hoàn toàn bằng NumPy.
    
    Hàm này đọc thông tin về xe và khách hàng, trả về dữ liệu dưới dạng
    các mảng NumPy hiệu năng cao.
    
    Returns:
        tuple: Một tuple chứa:
            - vehicle_capacity (int): Tải trọng của xe.
            - customer_data (np.ndarray): Mảng 2D chứa TOÀN BỘ dữ liệu của khách hàng.
            - cord_data (np.ndarray): Mảng 2D chỉ chứa tọa độ (X, Y) của khách hàng,
                                      sẵn sàng cho các thuật toán clustering như K-Means.
    """

    path = os.path.join(url, name_of_id)
    with open(path, 'r') as file:
        lines = file.readlines()

    num_vehicles, vehicle_capacity = map(int, lines[4].strip().split())

    customers = np.loadtxt(path, skiprows=9, usecols=(0, 1, 2, 3, 4, 5, 6), dtype=np.int64)
    cord_data = customers[:, C_X:C_Y+1]

    return vehicle_capacity, cord_data, customers

# print(load_txt_dataset(url="data/txt/100/R1/",name_of_id="R101.txt"))
