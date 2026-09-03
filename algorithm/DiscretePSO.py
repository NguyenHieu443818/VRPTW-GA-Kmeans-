"""
Discrete Particle Swarm Optimization (Discrete PSO) cho bài toán VRPTW
=======================================================================
Rời rạc hóa thuật toán PSO bằng cách định nghĩa lại:
  - Vị trí (Position)  : hoán vị (permutation) danh sách khách hàng
  - Vận tốc (Velocity) : danh sách các Swap Operator SO(a_i, a_j)
  - Phép cộng X + V    : áp dụng tuần tự chuỗi SO lên hoán vị
  - Phép trừ A - B     : tìm chuỗi SO biến đổi B thành A
  - Phép nhân r * V    : lọc ngẫu nhiên SO với xác suất r
  - Phép ghép V1 ⊕ V2  : nối hai chuỗi SO

Tham khảo: Discrete PSO cho TSP/VRPTW — công thức (17)–(21) trong nghiên cứu.

Tính toán fitness dùng chung qua `ultility.vrptw_evaluator` (numba JIT).

Cấu trúc dữ liệu đầu vào:
    customers : np.ndarray shape (n+1, 7)
        - Hàng 0  : depot
        - Cột     : [ID, X, Y, demand, ready_time, due_time, service_time]
    graph_data: np.ndarray shape (n+1, n+1)
        - Ma trận khoảng cách Euclidean giữa các điểm
"""

import numpy as np
import random
import time
from ultility.vrptw_evaluator import (
    evaluate_route,
    individual_to_route as _ito_fn,
    re_cluster_by_timewindow as _re_cluster_fn,
)


# ======================================================================
# Lớp SwapOperator: đại diện cho một phép hoán đổi SO(pos_a, pos_b)
# ======================================================================
class SwapOperator:
    """
    Toán tử hoán đổi SO(pos_a, pos_b): tráo đổi phần tử
    tại hai vị trí pos_a và pos_b trong chuỗi hoán vị.

    Parameters
    ----------
    pos_a : int  – vị trí đầu tiên (chỉ số trong chuỗi)
    pos_b : int  – vị trí thứ hai  (chỉ số trong chuỗi)
    """
    __slots__ = ('pos_a', 'pos_b')

    def __init__(self, pos_a: int, pos_b: int):
        self.pos_a = pos_a
        self.pos_b = pos_b

    def apply(self, sequence: list) -> list:
        """Áp dụng phép hoán đổi lên chuỗi, trả về chuỗi mới."""
        seq = sequence[:]
        seq[self.pos_a], seq[self.pos_b] = seq[self.pos_b], seq[self.pos_a]
        return seq

    def __repr__(self):
        return f"SO({self.pos_a},{self.pos_b})"


# ======================================================================
# Lớp DiscreteParticle: đại diện cho một hạt trong bầy đàn rời rạc
# ======================================================================
class DiscreteParticle:
    """
    Một hạt (particle) trong không gian rời rạc VRPTW.

    Attributes
    ----------
    position       : list[int]  – hoán vị hiện tại (1-indexed)
    velocity       : list[SwapOperator]  – chuỗi SO hiện tại
    pbest_position : list[int]  – hoán vị tốt nhất cá nhân
    pbest_fitness  : float
    pbest_distance : float
    """

    def __init__(self, cluster: list):
        self.position: list        = random.sample(cluster, len(cluster))
        self.velocity: list        = []
        self.pbest_position: list  = self.position[:]
        self.pbest_fitness: float  = float('inf')
        self.pbest_distance: float = float('inf')


# ======================================================================
# Lớp DiscretePSO: thuật toán chính
# ======================================================================
class DiscretePSO:
    """
    Discrete Particle Swarm Optimization cho bài toán VRPTW.

    Toán tử rời rạc:
      • X + V   : áp dụng tuần tự SO trong V lên X
      • A - B   : chuỗi SO biến đổi B thành A (selection-sort based)
      • r * V   : giữ mỗi SO với xác suất r
      • V1 ⊕ V2 : ghép hai chuỗi SO

    Đánh giá fitness dùng chung qua vrptw_evaluator (numba JIT).

    Parameters
    ----------
    num_particles    : int   – số hạt trong bầy đàn
    max_iter         : int   – số vòng lặp tối đa
    vehicle_capacity : float – tải trọng tối đa xe
    M                : float – sai số time-window
    w                : float – trọng số quán tính (xác suất giữ SO cũ)
    c1               : float – hệ số học cá nhân
    c2               : float – hệ số học xã hội
    customers        : np.ndarray shape (n+1, 7)
    graph_data       : np.ndarray shape (n+1, n+1)
    sditer           : int   – Dissolution Rule: số thế hệ không cải thiện
    """

    def __init__(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        vehicle_capacity: float = 200,
        M: float = 0,
        w: float = 0.8,
        c1: float = 0.5,
        c2: float = 0.5,
        customers: np.ndarray = None,
        graph_data: np.ndarray = None,
        sditer: int = 50,
    ):
        self.num_particles    = num_particles
        self.max_iter         = max_iter
        self._vehicle_capacity = vehicle_capacity
        self._M               = M
        self.w                = w
        self.c1               = c1
        self.c2               = c2
        self.customers        = customers
        self.graph_data       = graph_data
        self.sditer           = sditer

        # Kết quả tích lũy sau fit()
        self.best_fitness_global: float  = 0.0
        self.best_distance_global: float = 0.0
        self.route_count_global: int     = 0
        self.best_route_global: list     = []
        self.process_time: float         = 0.0

    # ==================================================================
    # Phần I: Các toán tử rời rạc cốt lõi
    # ==================================================================

    @staticmethod
    def _position_add_velocity(position: list, velocity: list) -> list:
        """X' = X + V — áp dụng tuần tự tất cả SO trong V lên hoán vị X."""
        result = position[:]
        for so in velocity:
            result = so.apply(result)
        return result

    @staticmethod
    def _position_subtract(A: list, B: list) -> list:
        """
        V = A - B
        Trả về chuỗi SO tối thiểu để biến đổi B thành A.
        Tính chất đảm bảo: B + (A - B) = A.
        """
        temp     = B[:]
        velocity = []
        for i, target in enumerate(A):
            j = temp.index(target)
            if i != j:
                velocity.append(SwapOperator(i, j))
                temp[i], temp[j] = temp[j], temp[i]
        return velocity

    @staticmethod
    def _scale_velocity(velocity: list, prob: float) -> list:
        """r * V — giữ mỗi SO với xác suất prob ∈ [0, 1]."""
        return [so for so in velocity if random.random() < prob]

    @staticmethod
    def _merge_velocities(*velocities) -> list:
        """V1 ⊕ V2 ⊕ ... — ghép nhiều chuỗi SO thành một."""
        merged = []
        for v in velocities:
            merged.extend(v)
        return merged

    # ==================================================================
    # Phần II: Đánh giá fitness (dùng numba JIT từ vrptw_evaluator)
    # ==================================================================

    def _evaluate(self, position: list) -> tuple:
        """
        Tính (fitness, distance) cho một hoán vị.
        Delegate sang evaluate_route() numba JIT.
        """
        return evaluate_route(
            np.asarray(position, dtype=np.int64),
            self.customers,
            self.graph_data,
            self._vehicle_capacity,
            self._M,
        )

    def _individual_to_route(self, position: list) -> list:
        """
        Chuyển hoán vị → danh sách sub-routes.
        Delegate sang individual_to_route() trong evaluator.
        """
        routes, _, _ = _ito_fn(
            position, self.customers, self.graph_data,
            self._vehicle_capacity, self._M,
        )
        return routes

    # ==================================================================
    # Phần III: Vòng lặp PSO chính cho một cụm
    # ==================================================================

    def _fit_single_cluster(self, cluster: list) -> tuple:
        """
        Chạy Discrete PSO cho một cụm đơn.

        Returns
        -------
        (gbest_position, gbest_fitness, gbest_distance)
        """
        if not cluster:
            return [], 0.0, 0.0

        # Bước 1: Khởi tạo bầy đàn
        swarm = [DiscreteParticle(cluster) for _ in range(self.num_particles)]
        for particle in swarm:
            f, d = self._evaluate(particle.position)
            particle.pbest_fitness   = f
            particle.pbest_distance  = d

        best_p         = min(swarm, key=lambda p: p.pbest_fitness)
        gbest_position = best_p.pbest_position[:]
        gbest_fitness  = best_p.pbest_fitness
        gbest_distance = best_p.pbest_distance

        no_improve_count   = 0
        prev_gbest_fitness = gbest_fitness

        # Bước 2: Vòng lặp chính
        for _ in range(self.max_iter):
            for particle in swarm:
                r1 = random.random()
                r2 = random.random()

                # Tính các thành phần vận tốc
                inertia   = self._scale_velocity(particle.velocity, self.w)
                cognitive = self._scale_velocity(
                    self._position_subtract(particle.pbest_position, particle.position),
                    self.c1 * r1,
                )
                social = self._scale_velocity(
                    self._position_subtract(gbest_position, particle.position),
                    self.c2 * r2,
                )

                # Ghép vận tốc mới (⊕)
                particle.velocity = self._merge_velocities(inertia, cognitive, social)

                # Cập nhật vị trí: X(k+1) = X(k) + V(k+1)
                particle.position = self._position_add_velocity(
                    particle.position, particle.velocity)

                # Đánh giá và cập nhật pbest
                f, d = self._evaluate(particle.position)
                if f < particle.pbest_fitness:
                    particle.pbest_fitness   = f
                    particle.pbest_distance  = d
                    particle.pbest_position  = particle.position[:]

            # Cập nhật gbest
            best_in_gen = min(swarm, key=lambda p: p.pbest_fitness)
            if best_in_gen.pbest_fitness < gbest_fitness:
                gbest_fitness  = best_in_gen.pbest_fitness
                gbest_distance = best_in_gen.pbest_distance
                gbest_position = best_in_gen.pbest_position[:]

            # Dissolution Rule: dừng sớm nếu không cải thiện
            if gbest_fitness < prev_gbest_fitness:
                no_improve_count   = 0
                prev_gbest_fitness = gbest_fitness
            else:
                no_improve_count += 1

            if no_improve_count >= self.sditer:
                break

        return gbest_position, gbest_fitness, gbest_distance

    # ==================================================================
    # Phần IV: Giao diện công khai — tương thích với GA/GA_VNS
    # ==================================================================

    def fit(self, clusters: list) -> tuple:
        """
        Chạy Discrete PSO trên toàn bộ tập cụm.

        Parameters
        ----------
        clusters : list of list of int  – mỗi cụm là list 1-indexed

        Returns
        -------
        (best_fitness_global, best_route_global, best_distance_global,
         route_count_global, process_time)
        """
        _start = time.time()

        for cluster in clusters:
            if not cluster:
                continue
            gbest_pos, gbest_fit, gbest_dist = self._fit_single_cluster(cluster)

            self.best_fitness_global  += gbest_fit
            self.best_distance_global += gbest_dist
            routes = self._individual_to_route(gbest_pos)
            self.best_route_global.append(routes)
            self.route_count_global  += len(routes)

        self.process_time = time.time() - _start
        return (
            self.best_fitness_global,
            self.best_route_global,
            self.best_distance_global,
            self.route_count_global,
            self.process_time,
        )
