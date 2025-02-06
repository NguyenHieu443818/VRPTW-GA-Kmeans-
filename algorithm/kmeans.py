import numpy as np
from ultility.utilities import distance_cdist

# Thuật toán k-means để phân cụm dữ liệu làm các rout lộ trình


class Kmeans():
    def __init__(self, epsilon: float = 1e-5, maxiter: int = 1000, n_cluster: int = 3):
        self.__epsilon = epsilon  # Sai số dừng - epsilon
        self.__maxiter = maxiter  # Số vòng lặp
        self.n_cluster = n_cluster

    def __init__cluster_center(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed)
        return X[np.random.choice(X.shape[0], self.n_cluster, replace=False)]

    def update_cluster_center(self, X: np.ndarray, label: np.array) -> np.ndarray:
        cluster2 = [np.mean(X[label == i], axis=0)
                    for i in range(self.n_cluster)]
        return cluster2

    def update_membership_matrix(self, X: np.ndarray, V: np.ndarray) -> np.array:
        # e = [np.argmin(np.linalg.norm(data_point-V,axis=1))for data_point in X]
        e = np.argmin(distance_cdist(X, V), axis=1)
        return np.array(e)

    def label_to_cluster(self, labels: np.ndarray):
        return [np.argwhere([labels == i]).T[1,] for i in range(self.n_cluster)]

    def has_converged(self, centers, new_centers):
        # trả về True nếu hai tập hợp tâm cụm là giống nhau
        return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

    # X: ma trận điểm dữ liệu
    # C: số cụm
    def k_means(self, X: np.ndarray, seed: int = 42) -> tuple:

        # khởi tạo ma trận tâm cụm ngẫu nhiên
        v = self.__init__cluster_center(X, seed)
        for step in range(self.__maxiter):
            # cập nhật ma trận độ thuộc
            old_v = v.copy()
            u = self.update_membership_matrix(X, old_v)
            v = self.update_cluster_center(X, u)
            if self.has_converged(old_v, v):
                break
        return u, v, step + 1
    
    def k_means_sorted(self, X: np.ndarray,FirstPoint: np.ndarray, seed: int = 42,) -> tuple:

        u, v, step = self.k_means(X=X,seed=seed)
        v = self.sort_cluster_by_len(V=v,FirstPoint=FirstPoint)
        u = self.update_membership_matrix(X, v)
        
        return u, v, step + 1
    
    def k_means_lib_sorted(self, X: np.ndarray,FirstPoint: np.ndarray, seed: int = 42,) -> tuple:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=seed).fit(X)
        centroid  = kmeans.cluster_centers_
        v = self.sort_cluster_by_distance(V=centroid,FirstPoint=FirstPoint)
        u = self.update_membership_matrix(X, v)
        
        return u, v, 1

    def data_to_cluster(self, U: np.ndarray):
        return [np.argwhere([U == i]).T[1,] +1 for i in range(self.n_cluster)]
    
    # def sort_cluster(self,V: np.ndarray,FirstPoint: np.ndarray): # Sắp xếp lại danh sách các tâm cụm tăng dần từ điểm dữ liệu đầu vào đến các cụm
    #     distances = np.linalg.norm(FirstPoint - V, axis=1)
    #     sorted_indices = np.argsort(distances)
    #     # sorted_V = V[sorted_indices]
    #     sorted_V = np.zeros((len(V),2))

    #     for idx,val in enumerate(sorted_indices): 
    #         sorted_V[idx] = V[val]


    #     return sorted_V
    
    def sort_cluster_by_distance(self,V: np.ndarray, FirstPoint: np.ndarray) -> np.ndarray:
        # Danh sách để lưu trữ các điểm đã sắp xếp
        sorted_V = []
        
        # Danh sách các chỉ số đã được sử dụng để theo dõi điểm đã sắp xếp
        used_indices = set()
        
        # Bắt đầu từ FirstPoint
        current_point = FirstPoint
        
        for _ in range(len(V)):
            # Tính khoảng cách từ current_point đến tất cả các điểm trong V
            distances = np.linalg.norm(current_point - V, axis=1)
            
            # Chỉ lấy các chỉ số mà chưa được sử dụng
            valid_indices = [i for i in range(len(V)) if i not in used_indices]
            
            # Tìm chỉ số của điểm gần nhất trong các chỉ số hợp lệ
            nearest_index = valid_indices[np.argmin(distances[valid_indices])]
            
            # Thêm điểm gần nhất vào danh sách đã sắp xếp
            sorted_V.append(V[nearest_index])
            
            # Cập nhật current_point cho lần lặp tiếp theo
            current_point = V[nearest_index]
            
            # Đánh dấu chỉ số đã sử dụng
            used_indices.add(nearest_index)

        return np.array(sorted_V)

    def sort_cluster_by_len(self,V: np.ndarray, FirstPoint: np.ndarray) -> np.ndarray:
        V_len = [len(v) for v in V]
        sorted_indices = np.argsort(V_len)
        # sorted_V = V[sorted_indices]
        sorted_V = np.zeros((len(V),2))

        for idx,val in enumerate(sorted_indices): 
            sorted_V[idx] = V[val]

        return sorted_V


    def elbow_k_means(self, X: np.ndarray, C_scope):
        import matplotlib.pyplot as plt
        distortions = []
        for k in C_scope:
            self.n_cluster = k
            u, v, step = self.k_means(X=X)
            distortions.append(np.sum(np.min(distance_cdist(X, v), axis=1)**2))
        plt.figure(figsize=(10, 10))
        plt.plot(C_scope, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortions')
        plt.grid(True)
        plt.show()

    def elbow_k_means_lib(self, X: np.ndarray, C_scope):
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        # from sklearn.datasets import make_blobs

        # Danh sách để lưu trữ giá trị inertia
        inertia = []

        # Thử các giá trị k từ 1 đến 10
        for k in C_scope:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Vẽ đồ thị
        plt.plot(C_scope, inertia)
        plt.xlabel('Số lượng cụm (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.show()
