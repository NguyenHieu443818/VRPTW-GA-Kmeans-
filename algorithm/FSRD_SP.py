"""
FSRD-SP Algorithm: Fuzzy STD Rho-overlapping Decomposition with Route-pool Set Partitioning
==========================================================================================
Thuật toán VRPTW kết hợp phân rã cụm mờ chồng lấn, sinh route ứng viên
bằng GA giant-tour và chọn nghiệm toàn cục bằng Set Partitioning (SP).

Quy trình:
  1. Nhận các subproblem chồng lấn C_p từ bước phân cụm Fuzzy c-Medoids.
  2. Khởi tạo Route Pool Omega và nạp singleton routes [0, i, 0] để đảm bảo luôn khả thi.
  3. Chạy GA giant-tour trên từng subproblem C_p, giải mã và thu thập các sub-route vào Pool Omega.
  4. Giải bài toán Set Partitioning trên Pool Omega (MILP + Greedy fallback) để chọn ra
     tập route tối ưu toàn cục phủ chính xác 100% khách hàng đúng 1 lần.

Tham khảo: PROPOSED_MODEL_revised_updated.md
"""

from __future__ import annotations

import time
import numpy as np
from algorithm.GeneticAlgorithm import GA
from algorithm.SetPartitioning import RoutePool, SetPartitioningSolver
from ultility.utilities import round_float


class FSRD_SP(GA):
    """
    Thuật toán FSRD-SP (Fuzzy STD Rho-overlapping Decomposition with Route-pool Set Partitioning).
    """

    def __init__(
        self,
        individual: int = 150,
        generation: int = 150,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        vehicle_capacity: float = 200,
        conserve_rate: float = 0.1,
        M: float = 0,
        customers: np.ndarray = None,
        graph_data: np.ndarray = None,
    ):
        super().__init__(
            individual=individual,
            generation=generation,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            vehicle_capacity=vehicle_capacity,
            conserve_rate=conserve_rate,
            M=M,
            customers=customers,
            graph_data=graph_data,
        )

    def fit(self, clusters: list[list[int]]) -> tuple[float, list, float, int, float]:
        """
        Thực thi thuật toán FSRD-SP trên tập subproblem clusters (C_p).

        Parameters
        ----------
        clusters : list of list of int
            Danh sách các subproblem (chồng lấn hoặc rời rạc). Mỗi subproblem
            là list các chỉ số 1-indexed của khách hàng.

        Returns
        -------
        (best_fitness, best_route_global, best_distance, route_count, process_time)
        """
        start_time = time.time()
        n_customers = len(self.customers) - 1

        # 1. Khởi tạo Route Pool Omega và nạp singleton routes [0, i, 0]
        pool = RoutePool(
            customers=self.customers,
            graph_data=self.graph_data,
            vehicle_capacity=self._vehicle_capacity,
            M=self._M,
        )
        pool.add_singleton_routes(n_customers)

        # 2. Chạy GA giant-tour trên từng subproblem C_p và gom sub-routes vào Pool
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            if len(cluster) == 1:
                pool.add_route(cluster)
                continue

            # Khởi tạo quần thể cho subproblem C_p
            self.initial_population(cluster)
            self.cal_fitness_population()
            pool.collect_from_population(self.population)

            for _ in range(self._generation):
                self.selection()
                self.hybird()
                self.cal_fitness_population()
                pool.collect_from_population(self.population)

        # 3. Giải bài toán Set Partitioning Master Problem trên Route Pool
        best_routes, best_fitness, best_distance, route_count = (
            SetPartitioningSolver.solve(pool, n_customers)
        )

        process_time = round_float(time.time() - start_time)
        self.process_time = process_time
        self.best_fitness_global = best_fitness
        self.best_distance_global = best_distance
        self.route_count_global = route_count
        # Định dạng dạng list các nhóm route tương thích với main.py
        self.best_route_global = [best_routes]

        return (
            self.best_fitness_global,
            self.best_route_global,
            self.best_distance_global,
            self.route_count_global,
            self.process_time,
        )
