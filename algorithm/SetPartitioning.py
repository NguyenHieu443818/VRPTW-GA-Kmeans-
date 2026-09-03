"""
Set Partitioning & Route Pool for VRPTW (FSRD-SP Model)
======================================================
Quản lý tập hợp các route ứng viên (Route Pool Omega) và giải
bài toán Set Partitioning Master Problem để chọn ra tập route toàn cục
tối ưu phủ chính xác mỗi khách hàng đúng 1 lần.

Tham khảo: Section 6 trong PROPOSED_MODEL_revised_updated.md
"""

from __future__ import annotations

import numpy as np
from ultility.vrptw_evaluator import evaluate_route, analyze_route


class RoutePool:
    """
    Quản lý tập hợp các route ứng viên Omega cho bài toán Set Partitioning.

    - Khóa định danh mỗi route là tuple các khách hàng: (c1, c2, ...).
    - Tự động loại bỏ trùng lặp và chỉ lưu bản có chi phí c_r nhỏ nhất.
    - Đảm bảo tính khả thi bằng cách tự động bổ sung các singleton routes [0, i, 0].
    """

    def __init__(
        self,
        customers: np.ndarray,
        graph_data: np.ndarray,
        vehicle_capacity: float,
        M: float = 0.0,
    ):
        self.customers = customers
        self.graph_data = graph_data
        self.vehicle_capacity = vehicle_capacity
        self.M = M

        # routes_dict: tuple(customers) -> (fitness, distance)
        self._routes: dict[tuple[int, ...], tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self._routes)

    def add_route(self, route: list[int] | np.ndarray) -> bool:
        """
        Thêm một sub-route vào pool nếu hợp lệ.
        Nếu route đã tồn tại, giữ lại bản có fitness nhỏ hơn.
        """
        if len(route) == 0:
            return False

        route_tuple = tuple(int(x) for x in route)
        route_np = np.asarray(route_tuple, dtype=np.int64)

        # Tính chi phí route c_r = D(r) + W(r) + P(r) và khoảng cách D(r)
        fitness, distance = evaluate_route(
            route_np,
            self.customers,
            self.graph_data,
            self.vehicle_capacity,
            self.M,
        )

        if route_tuple in self._routes:
            if fitness < self._routes[route_tuple][0]:
                self._routes[route_tuple] = (float(fitness), float(distance))
                return True
            return False
        else:
            self._routes[route_tuple] = (float(fitness), float(distance))
            return True

    def add_singleton_routes(self, n_customers: int):
        """
        Thêm route đơn lẻ [0, i, 0] cho mọi khách hàng i = 1..n.
        Đảm bảo Set Partitioning luôn có nghiệm khả thi phủ 100% khách hàng.
        """
        for cust_id in range(1, n_customers + 1):
            self.add_route([cust_id])

    def collect_from_giant_tour(self, giant_tour: list[int] | np.ndarray):
        """
        Tách một giant-tour thành các sub-route hợp lệ và thêm vào pool.
        """
        if len(giant_tour) == 0:
            return

        tour_np = np.asarray(giant_tour, dtype=np.int64)
        split_points, _, _ = analyze_route(
            tour_np,
            self.customers,
            self.graph_data,
            self.vehicle_capacity,
            self.M,
        )

        last = 0
        for sp in split_points:
            sub = giant_tour[last:sp]
            if len(sub) > 0:
                self.add_route(sub)
            last = sp
        final = giant_tour[last:]
        if len(final) > 0:
            self.add_route(final)

    def collect_from_population(self, population):
        """
        Thu thập tất cả sub-routes từ một quần thể cá thể GA (hoặc danh sách giant tours).
        """
        for ind in population:
            if hasattr(ind, "customer_list"):
                self.collect_from_giant_tour(ind.customer_list)
            elif isinstance(ind, (list, np.ndarray)):
                self.collect_from_giant_tour(ind)

    def get_routes_and_costs(self) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """
        Trả về (danh_sách_routes, mảng_chi_phí_fitness, mảng_khoảng_cách).
        """
        routes_list = [list(k) for k in self._routes.keys()]
        costs = np.array([v[0] for v in self._routes.values()], dtype=np.float64)
        distances = np.array([v[1] for v in self._routes.values()], dtype=np.float64)
        return routes_list, costs, distances


class SetPartitioningSolver:
    """
    Bộ giải bài toán Set Partitioning Master Problem cho VRPTW.

    Mô hình toán:
        minimize    sum_{r in Omega} c_r * x_r
        subject to  sum_{r: i in r} x_r = 1    với mọi i in C
                    sum_{r in Omega} x_r <= K  (tùy chọn)
                    x_r in {0, 1}

    Chiến lược giải:
      1. Phương pháp chính: MILP qua `scipy.optimize.milp` (nhanh, chuẩn xác).
      2. Phương pháp dự phòng: Greedy Set Partitioning (luôn đảm bảo trả về nghiệm).
    """

    @staticmethod
    def solve(
        pool: RoutePool,
        n_customers: int,
        max_vehicles: int | None = None,
    ) -> tuple[list[list[int]], float, float, int]:
        """
        Giải Set Partitioning trên RoutePool.

        Parameters
        ----------
        pool : RoutePool
            Tập route ứng viên đã thu thập.
        n_customers : int
            Tổng số khách hàng (1-indexed, từ 1 đến n_customers).
        max_vehicles : int, optional
            Số lượng xe tối đa cho phép.

        Returns
        -------
        (best_routes, best_fitness, best_distance, route_count)
        """
        routes_list, costs, distances = pool.get_routes_and_costs()
        n_routes = len(routes_list)

        if n_routes == 0:
            raise RuntimeError("RoutePool rỗng, không thể giải Set Partitioning.")

        # Thử giải bằng MILP (scipy.optimize.milp)
        try:
            sol_routes, sol_fitness, sol_distance = SetPartitioningSolver._solve_milp(
                routes_list, costs, distances, n_customers, max_vehicles
            )
            if sol_routes is not None:
                return sol_routes, sol_fitness, sol_distance, len(sol_routes)
        except Exception:
            pass

        # Fallback sang Greedy Set Partitioning
        sol_routes, sol_fitness, sol_distance = SetPartitioningSolver._solve_greedy(
            routes_list, costs, distances, n_customers
        )
        return sol_routes, sol_fitness, sol_distance, len(sol_routes)

    @staticmethod
    def _solve_milp(
        routes_list: list[list[int]],
        costs: np.ndarray,
        distances: np.ndarray,
        n_customers: int,
        max_vehicles: int | None = None,
    ) -> tuple[list[list[int]] | None, float, float]:
        """
        Giải Set Partitioning bằng scipy.optimize.milp.
        """
        from scipy.optimize import milp, LinearConstraint
        from scipy.sparse import dok_matrix

        n_routes = len(routes_list)

        # Xây dựng ma trận thưa A: shape (n_customers, n_routes)
        # A[i-1, r] = 1 nếu khách hàng i nằm trong route r
        A = dok_matrix((n_customers, n_routes), dtype=np.float64)
        for r_idx, route in enumerate(routes_list):
            for cust in route:
                if 1 <= cust <= n_customers:
                    A[cust - 1, r_idx] = 1.0

        A_csc = A.tocsc()

        # Ràng buộc đẳng thức: mỗi khách hàng được phục vụ đúng 1 lần (A * x == 1)
        lhs = np.ones(n_customers, dtype=np.float64)
        rhs = np.ones(n_customers, dtype=np.float64)
        constraints = [LinearConstraint(A_csc, lhs, rhs)]

        # Ràng buộc số lượng xe nếu có: sum(x_r) <= max_vehicles
        if max_vehicles is not None and max_vehicles > 0:
            ones_row = np.ones((1, n_routes), dtype=np.float64)
            constraints.append(
                LinearConstraint(ones_row, lb=0.0, ub=float(max_vehicles))
            )

        # Biến nhị phân x_r in {0, 1}
        integrality = np.ones(n_routes, dtype=np.int64)  # 1 = integer

        res = milp(
            c=costs,
            integrality=integrality,
            constraints=constraints,
            bounds=(0.0, 1.0),
        )

        if res.success and res.x is not None:
            chosen_indices = np.where(res.x > 0.5)[0]

            # Xác nhận nghiệm phủ chính xác 100%
            covered = np.zeros(n_customers, dtype=bool)
            valid = True
            for idx in chosen_indices:
                for c in routes_list[idx]:
                    if covered[c - 1]:
                        valid = False
                        break
                    covered[c - 1] = True
                if not valid:
                    break

            if valid and covered.all():
                chosen_routes = [routes_list[i] for i in chosen_indices]
                total_fitness = float(np.sum(costs[chosen_indices]))
                total_distance = float(np.sum(distances[chosen_indices]))
                return chosen_routes, total_fitness, total_distance

        return None, 0.0, 0.0

    @staticmethod
    def _solve_greedy(
        routes_list: list[list[int]],
        costs: np.ndarray,
        distances: np.ndarray,
        n_customers: int,
    ) -> tuple[list[list[int]], float, float]:
        """
        Thuật toán Greedy Set Partitioning (Fallback):
        Lần lượt chọn các route có chi phí đơn vị (c_r / len(r)) tốt nhất mà
        chỉ chứa các khách hàng chưa được phục vụ (r subseteq uncovered).
        """
        uncovered = set(range(1, n_customers + 1))
        chosen_routes: list[list[int]] = []
        total_fitness = 0.0
        total_distance = 0.0

        # Tính tỷ lệ chi phí trên mỗi khách hàng
        route_lens = np.array([max(len(r), 1) for r in routes_list], dtype=np.float64)
        efficiency = costs / route_lens

        # Sắp xếp chỉ số route theo efficiency tăng dần
        sorted_indices = np.argsort(efficiency)

        for r_idx in sorted_indices:
            route = routes_list[r_idx]
            route_set = set(route)

            # Chỉ chọn nếu toàn bộ khách trong route đều chưa được phục vụ
            if route_set.issubset(uncovered):
                chosen_routes.append(route)
                total_fitness += costs[r_idx]
                total_distance += distances[r_idx]
                uncovered -= route_set

                if not uncovered:
                    break

        # Nếu còn khách hàng sót, tìm các singleton routes tương ứng
        if uncovered:
            singleton_map = {
                r[0]: (costs[i], distances[i])
                for i, r in enumerate(routes_list)
                if len(r) == 1
            }
            for cust in sorted(uncovered):
                chosen_routes.append([cust])
                if cust in singleton_map:
                    total_fitness += singleton_map[cust][0]
                    total_distance += singleton_map[cust][1]

        return chosen_routes, total_fitness, total_distance
