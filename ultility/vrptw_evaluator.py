"""
VRPTW Evaluator — Module tính toán hàm mục tiêu dùng chung
===========================================================
Tập trung toàn bộ logic đánh giá giải pháp VRPTW vào một nơi,
được tối ưu bằng Numba JIT để dùng chung cho GA, DiscretePSO và các
thuật toán khác.

Hàm xuất khẩu (public API):
    evaluate_route()        – tính (fitness, distance) cho 1 hoán vị
    analyze_route()         – tách + tính chi tiết từng sub-route
    evaluate_batch()        – tính fitness toàn bộ population (song song)
    greedy_insert()         – chèn gen vào vị trí tối ưu (numba)
    individual_to_route()   – Python wrapper: hoán vị → danh sách route
    re_cluster_by_timewindow() – gộp cụm theo ràng buộc time-window

Quy ước cột trong mảng customers (np.ndarray shape (n+1, 7)):
    [0]=ID  [1]=X  [2]=Y  [3]=demand  [4]=ready_time  [5]=due_time  [6]=service_time
"""

import numpy as np
import numba

# ─── Chỉ số cột (dùng chung với các thuật toán) ──────────────────────────────
C_ID           = 0
C_X            = 1
C_Y            = 2
C_DEMAND       = 3
C_READY_TIME   = 4
C_DUE_TIME     = 5
C_SERVICE_TIME = 6


# ═════════════════════════════════════════════════════════════════════════════
# Hàm Numba JIT — biên dịch 1 lần, tái sử dụng không giới hạn
# ═════════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def evaluate_route(
    individual: np.ndarray,
    customers: np.ndarray,
    graph_data: np.ndarray,
    vehicle_capacity: float,
    M: float,
    fitness_bound: float = np.inf,
) -> tuple:
    """
    Tính (fitness, distance) cho một hoán vị khách hàng.

    Fitness = tổng khoảng cách + tổng thời gian chờ & trễ (penalty).
    Hỗ trợ branch-and-bound cắt sớm qua `fitness_bound`.

    Parameters
    ----------
    individual      : np.ndarray[int64] – hoán vị 1-indexed của khách hàng
    customers       : np.ndarray[int64] shape (n+1, 7) – depot ở hàng 0
    graph_data      : np.ndarray[float64] shape (n+1, n+1) – ma trận khoảng cách
    vehicle_capacity: float – tải trọng tối đa xe
    M               : float – sai số cho phép trong time-window
    fitness_bound   : float – dừng sớm nếu fitness vượt ngưỡng này

    Returns
    -------
    (fitness: float, distance: float)
    """
    depot_ready  = customers[0, C_READY_TIME]
    depot_due    = customers[0, C_DUE_TIME] + M
    vehicle_load = 0.0
    elapsed_time = depot_ready
    last_id      = np.int64(0)
    fitness      = 0.0
    distance     = 0.0

    for cust_id in individual:
        demand       = customers[cust_id, C_DEMAND]
        ready_time   = customers[cust_id, C_READY_TIME]
        due_time     = customers[cust_id, C_DUE_TIME]
        service_time = customers[cust_id, C_SERVICE_TIME]

        moving_time    = graph_data[cust_id, last_id]
        arrive_time    = moving_time + elapsed_time
        waiting_time   = max(ready_time - M - arrive_time, 0.0)
        delay_time     = max(arrive_time - due_time - M, 0.0)
        departure_time = arrive_time + waiting_time + service_time
        return_time    = graph_data[cust_id, 0]

        update_load         = vehicle_load + demand
        total_time_if_return = departure_time + return_time

        if (update_load <= vehicle_capacity) and (total_time_if_return <= depot_due):
            # Tiếp tục trên xe hiện tại
            vehicle_load = update_load
            elapsed_time = departure_time
            distance    += moving_time
            fitness     += waiting_time + delay_time
        else:
            # Xe mới: đóng sub-route hiện tại và xuất phát từ depot
            distance += graph_data[last_id, 0]

            travel_from_depot = graph_data[0, cust_id]
            distance       += travel_from_depot
            arrive_new      = depot_ready + travel_from_depot
            waiting_time    = max(ready_time - M - arrive_new, 0.0)
            elapsed_time    = arrive_new + waiting_time + service_time
            fitness        += waiting_time
            vehicle_load    = demand

        last_id = cust_id

        # Branch-and-bound: dừng sớm nếu vượt ngưỡng
        if fitness > fitness_bound:
            return fitness, distance

    distance += graph_data[last_id, 0]
    fitness  += distance
    return fitness, distance


@numba.njit(cache=True)
def analyze_route(
    individual: np.ndarray,
    customers: np.ndarray,
    graph_data: np.ndarray,
    vehicle_capacity: float,
    M: float,
) -> tuple:
    """
    Tách hoán vị thành các sub-routes và tính chi tiết fitness/distance
    cho từng sub-route.

    Parameters
    ----------
    individual      : np.ndarray[int64]
    customers       : np.ndarray[int64] shape (n+1, 7)
    graph_data      : np.ndarray[float64] shape (n+1, n+1)
    vehicle_capacity: float
    M               : float

    Returns
    -------
    (split_indices, fitness_sub_routes, distance_sub_routes)
        - split_indices       : np.ndarray[int64]   – vị trí bắt đầu sub-route mới
        - fitness_sub_routes  : np.ndarray[float64] – fitness của từng sub-route
        - distance_sub_routes : np.ndarray[float64] – distance của từng sub-route
    """
    n        = len(individual)
    depot_ready = customers[0, C_READY_TIME]
    depot_due = customers[0, C_DUE_TIME] + M
    vehicle_load = 0.0
    elapsed_time = depot_ready
    last_id      = np.int64(0)

    split_indices       = np.empty(n,     dtype=np.int64)
    fitness_sub_routes  = np.empty(n + 1, dtype=np.float64)
    distance_sub_routes = np.empty(n + 1, dtype=np.float64)
    n_splits = 0
    n_routes = 0
    fitness  = 0.0
    distance = 0.0

    for i, cust_id in enumerate(individual):
        demand       = customers[cust_id, C_DEMAND]
        ready_time   = customers[cust_id, C_READY_TIME]
        due_time     = customers[cust_id, C_DUE_TIME]
        service_time = customers[cust_id, C_SERVICE_TIME]

        moving_time    = graph_data[cust_id, last_id]
        arrive_time    = moving_time + elapsed_time
        waiting_time   = max(ready_time - M - arrive_time, 0.0)
        delay_time     = max(arrive_time - due_time - M, 0.0)
        departure_time = arrive_time + waiting_time + service_time
        return_time    = graph_data[cust_id, 0]

        update_load         = vehicle_load + demand
        total_time_if_return = departure_time + return_time

        if (update_load <= vehicle_capacity) and (total_time_if_return <= depot_due):
            vehicle_load = update_load
            elapsed_time = departure_time
            distance    += moving_time
            fitness     += waiting_time + delay_time
        else:
            # Kết thúc sub-route hiện tại
            distance += graph_data[last_id, 0]
            fitness  += distance
            split_indices[n_splits]       = i
            distance_sub_routes[n_routes] = distance
            fitness_sub_routes[n_routes]  = fitness
            n_splits += 1
            n_routes += 1

            # Bắt đầu sub-route mới
            travel_from_depot = graph_data[0, cust_id]
            arrive_new      = depot_ready + travel_from_depot
            waiting_time    = max(ready_time - M - arrive_new, 0.0)
            elapsed_time    = arrive_new + waiting_time + service_time
            fitness         = waiting_time
            distance        = travel_from_depot
            vehicle_load    = demand

        last_id = cust_id

    # Đóng sub-route cuối
    distance += graph_data[last_id, 0]
    fitness  += distance
    distance_sub_routes[n_routes] = distance
    fitness_sub_routes[n_routes]  = fitness
    n_routes += 1

    return split_indices[:n_splits], fitness_sub_routes[:n_routes], distance_sub_routes[:n_routes]


@numba.njit(parallel=True, cache=True)
def evaluate_batch(
    population_2d: np.ndarray,
    customers: np.ndarray,
    graph_data: np.ndarray,
    vehicle_capacity: float,
    M: float,
) -> tuple:
    """
    Tính fitness cho toàn bộ population song song (numba.prange).

    Parameters
    ----------
    population_2d : np.ndarray[int64] shape (pop_size, n_customers)
    customers     : np.ndarray[int64] shape (n+1, 7)
    graph_data    : np.ndarray[float64] shape (n+1, n+1)
    vehicle_capacity: float
    M             : float

    Returns
    -------
    (fitness_arr, distance_arr) – mỗi array shape (pop_size,)
    """
    n            = population_2d.shape[0]
    fitness_arr  = np.empty(n, dtype=np.float64)
    distance_arr = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        fitness_arr[i], distance_arr[i] = evaluate_route(
            population_2d[i], customers, graph_data, vehicle_capacity, M)
    return fitness_arr, distance_arr


@numba.njit(cache=True)
def greedy_insert(
    base_gene: np.ndarray,
    diff_gene: np.ndarray,
    customers: np.ndarray,
    graph_data: np.ndarray,
    vehicle_capacity: float,
    M: float,
) -> np.ndarray:
    """
    Chèn từng phần tử của `diff_gene` vào `base_gene` tại vị trí có
    fitness tốt nhất (greedy insertion). Toàn bộ chạy trong numba.

    Parameters
    ----------
    base_gene : np.ndarray[int64] – chuỗi gen nền
    diff_gene : np.ndarray[int64] – chuỗi gen cần chèn

    Returns
    -------
    np.ndarray[int64] – chuỗi gen sau khi chèn
    """
    n_base = len(base_gene)
    n_diff = len(diff_gene)
    total  = n_base + n_diff

    result      = np.empty(total, dtype=np.int64)
    result[:n_base] = base_gene
    current_len = n_base
    buf         = np.empty(total, dtype=np.int64)

    for k in range(n_diff):
        gen          = diff_gene[k]
        n            = current_len
        best_idx     = np.int64(0)
        best_fitness = np.inf

        for idx in range(n + 1):
            buf[:idx]          = result[:idx]
            buf[idx]           = gen
            buf[idx + 1:n + 1] = result[idx:n]
            f, _ = evaluate_route(
                buf[:n + 1], customers, graph_data, vehicle_capacity, M, best_fitness)
            if f < best_fitness:
                best_fitness = f
                best_idx     = idx

        # In-place shift để chèn (tránh tạo array mới)
        for j in range(current_len, best_idx, -1):
            result[j] = result[j - 1]
        result[best_idx] = gen
        current_len += 1

    return result[:current_len]


# ═════════════════════════════════════════════════════════════════════════════
# Python wrappers — tiện dùng trong context Python thông thường
# ═════════════════════════════════════════════════════════════════════════════

def individual_to_route(
    individual,
    customers: np.ndarray,
    graph_data: np.ndarray,
    vehicle_capacity: float,
    M: float,
) -> tuple:
    """
    Chuyển hoán vị khách hàng → danh sách sub-routes (Python list).

    Parameters
    ----------
    individual : list or np.ndarray – hoán vị 1-indexed của khách hàng

    Returns
    -------
    (routes, fitness_subs, distance_subs)
        - routes         : list of (list or ndarray) – từng sub-route
        - fitness_subs   : np.ndarray[float64]
        - distance_subs  : np.ndarray[float64]
    """
    if len(individual) == 0:
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    individual_np = np.asarray(individual, dtype=np.int64)
    split_points, fitness_subs, distance_subs = analyze_route(
        individual_np, customers, graph_data, vehicle_capacity, M)

    routes = []
    last   = 0
    for sp in split_points:
        sub = individual[last:sp]
        if len(sub) > 0:
            routes.append(sub)
        last = sp
    final = individual[last:]
    if len(final) > 0:
        routes.append(final)

    return routes, fitness_subs, distance_subs


def re_cluster_by_timewindow(
    clusters: list,
    customers: np.ndarray,
    graph_data: np.ndarray,
    vehicle_capacity: float,
    M: float,
) -> list:
    """
    Gộp các cụm nhỏ nếu tổng nhu cầu và thời gian phục vụ cho phép
    cùng một xe hoàn thành trong cửa sổ thời gian depot.

    Parameters
    ----------
    clusters : list of list of int  – mỗi cụm là list chỉ số 1-indexed
    customers, graph_data, vehicle_capacity, M : như evaluate_route

    Returns
    -------
    list of list of int – danh sách cụm sau khi gộp
    """
    check_due = (float(customers[0, C_DUE_TIME])
                 - float(customers[0, C_READY_TIME]) + M)

    def _check_concat(c1: list, c2: list) -> bool:
        total = c1 + c2
        if np.sum(customers[total, C_DEMAND]) > vehicle_capacity:
            return False

        total_service = float(np.sum(customers[total, C_SERVICE_TIME]))
        dist_sub = graph_data[np.ix_(total, total)]
        nonzeros = dist_sub[dist_sub > 0]
        aver_dist = float(np.mean(nonzeros)) if len(nonzeros) > 0 else 0.0

        dist_to_depot = graph_data[0, total]
        depot_dists = dist_to_depot[dist_to_depot > 0]
        avg_dist_depot = float(np.mean(depot_dists)) if len(depot_dists) > 0 else 0.0

        total_move = 2.0 * avg_dist_depot + (len(total) - 1) * aver_dist
        return (total_service + total_move) <= check_due

    i = 0
    while i < len(clusters) - 1:
        j = i + 1
        while j < len(clusters):
            if _check_concat(clusters[i], clusters[j]):
                clusters = (
                    [clusters[i] + clusters[j]]
                    + [clusters[k] for k in range(len(clusters)) if k != i and k != j]
                )
                j = i + 1
            else:
                j += 1
        i += 1

    return clusters
