import numpy as np
from ultility.vrptw_evaluator import (
    evaluate_route, analyze_route, evaluate_batch,
    greedy_insert,
    individual_to_route as _ito_fn,
    re_cluster_by_timewindow as _re_cluster_fn,
)
import math
import random
from ultility.utilities import round_float
import time

C_ID = 0
C_X = 1
C_Y = 2
C_DEMAND = 3
C_READY_TIME = 4
C_DUE_TIME = 5
C_SERVICE_TIME = 6


class Individual():
    def __init__(self, customer_list=None, fitness: float = 0, distance: float = 0):
        self.customer_list = customer_list
        self.fitness = fitness
        self.distance = distance

    def print(self):
        print(self.customer_list, ' ', self.fitness, ' ', self.distance)


class GA:
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.15, vehicle_capacity: float = 200, conserve_rate: float = 0.1,
                 M: float = 0, customers: np.ndarray = None, graph_data: np.ndarray = None):
        self._individual = individual
        self._generation = generation
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._conserve_rate = conserve_rate

        self.population = []

        self._vehicle_capacity = vehicle_capacity
        self._M = M
        self.customers = customers
        self.graph_data = graph_data

        self.best_distance_global = 0
        self.route_count_global = 0
        self.best_fitness_global = 0
        self.best_route_global = []

        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

        self.process_time = 0

    def initial_population(self, cluster) -> list:
        # Lưu trực tiếp dưới dạng numpy int64 để tránh chuyển đổi lặp lại
        self.population = [
            Individual(customer_list=np.array(random.sample(cluster, len(cluster)), dtype=np.int64))
            for _ in range(self._individual)
        ]

    def individual_to_route(self, individual):
        """Chuyển hoán vị → danh sách sub-routes (dùng evaluator chung)."""
        if len(individual) == 0:
            return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        return _ito_fn(
            individual, self.customers, self.graph_data,
            self._vehicle_capacity, self._M
        )

    # Tách thành các lộ trình con (giữ lại cho backward compat — delegate sang analyze_route)
    @staticmethod
    def analyze_individual(individual: np.ndarray, customers: np.ndarray, graph_data: np.ndarray,
                           vehicle_capacity: float, M: float):
        """Wrapper tương thích ngược — delegate sang analyze_route của evaluator."""
        return analyze_route(individual, customers, graph_data, vehicle_capacity, M)

    @staticmethod
    def cal_fitness_individualV2(individual: np.ndarray, customers: np.ndarray, graph_data: np.ndarray,
                                  vehicle_capacity: float, M: float, fitness_to_branch_bound: float = np.inf):
        """Wrapper tương thích ngược — delegate sang evaluate_route của evaluator."""
        return evaluate_route(
            individual, customers, graph_data, vehicle_capacity, M, fitness_to_branch_bound)

    # Tính fitness toàn bộ quần thể bằng batch parallel (numba.prange)
    def cal_fitness_population(self):
        pop_2d = np.array([ind.customer_list for ind in self.population], dtype=np.int64)
        fitness_arr, dist_arr = evaluate_batch(
            pop_2d, self.customers, self.graph_data, self._vehicle_capacity, self._M)
        for i, ind in enumerate(self.population):
            ind.fitness  = fitness_arr[i]
            ind.distance = dist_arr[i]

    def selection(self):
        self.population.sort(key=lambda x: x.fitness)
        positionToDel = math.floor(self._individual * (1 + self._conserve_rate) / 2)
        del self.population[positionToDel:]

    def single_point_crossover(self, dad, mom):
        assert len(dad) == len(mom), "Dad and Mom must have the same length."

        pos1 = random.randrange(len(mom))

        mom_tail_set = set(mom[pos1:])
        filter_dad   = [gene for gene in dad if gene not in mom_tail_set]

        dad_tail_set = set(dad[pos1:])
        filter_mom   = [gene for gene in mom if gene not in dad_tail_set]

        gene_child_1 = filter_dad[:pos1] + mom[pos1:]
        gene_child_2 = filter_mom[:pos1] + dad[pos1:]

        return gene_child_1, gene_child_2

    def heuristic_single_point_crossover(self, dad, mom):
        sub_route_mom, fitness_sub_route_mom, _ = self.individual_to_route(mom)
        sub_route_dad, fitness_sub_route_dad, _ = self.individual_to_route(dad)

        # np.argmin thay vì min() + .index() (2 lần duyệt → 1 lần)
        idx_best_mom           = int(np.argmin(fitness_sub_route_mom))
        best_fitness_sub_route_mom = float(fitness_sub_route_mom[idx_best_mom])
        if self.best_fitness_pM <= best_fitness_sub_route_mom:
            pos1 = random.randrange(len(mom))
        else:
            best_sub_route_mom = sub_route_mom[idx_best_mom]
            pos1 = mom.index(best_sub_route_mom[0])

        mom_tail_set = set(mom[pos1:])
        filter_dad   = [gene for gene in dad if gene not in mom_tail_set]
        gene_child_1 = filter_dad[:pos1] + mom[pos1:]

        idx_best_dad           = int(np.argmin(fitness_sub_route_dad))
        best_fitness_sub_route_dad = float(fitness_sub_route_dad[idx_best_dad])
        if self.best_fitness_pD <= best_fitness_sub_route_dad:
            pos1 = random.randrange(len(dad))
        else:
            best_sub_route_dad = sub_route_dad[idx_best_dad]
            pos1 = dad.index(best_sub_route_dad[0])

        dad_tail_set = set(dad[pos1:])
        filter_mom   = [gene for gene in mom if gene not in dad_tail_set]
        gene_child_2 = filter_mom[:pos1] + dad[pos1:]

        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def two_point_crossover(self, dad, mom):
        sub_route_mom, _, _ = self.individual_to_route(mom)
        sub_route_dad, _, _ = self.individual_to_route(dad)

        pos1, pos2 = sorted(random.choices(range(len(sub_route_mom)), k=2))
        sub_mom = [item for sublist in sub_route_mom[pos1:pos2 + 1] for item in sublist]

        pos1, pos2 = sorted(random.choices(range(len(sub_route_dad)), k=2))
        sub_dad = [item for sublist in sub_route_dad[pos1:pos2 + 1] for item in sublist]

        # Dùng set để O(1) lookup thay vì O(N) list check
        sub_mom_set  = set(sub_mom)
        filter_dad   = [item for item in dad if item not in sub_mom_set]
        gene_child_1 = filter_dad[:pos1] + sub_mom + filter_dad[pos1:]

        sub_dad_set  = set(sub_dad)
        filter_mom   = [item for item in mom if item not in sub_dad_set]
        gene_child_2 = filter_mom[:pos1] + sub_dad + filter_mom[pos1:]

        return gene_child_1, gene_child_2

    def heuristic_two_point_crossover_v1(self, dad, mom):
        size = len(mom)
        if size < 2:
            return dad, mom

        sub_route_mom, fitness_sub_route_mom, _ = self.individual_to_route(mom)
        sub_route_dad, fitness_sub_route_dad, _ = self.individual_to_route(dad)

        idx_best_mom           = int(np.argmin(fitness_sub_route_mom))
        best_fitness_sub_route_mom = float(fitness_sub_route_mom[idx_best_mom])
        if self.best_fitness_pM <= best_fitness_sub_route_mom or len(sub_route_mom) == 0:
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_mom = sub_route_mom[idx_best_mom]
            pos1 = mom.index(best_sub_route_mom[0])
            pos2 = mom.index(best_sub_route_mom[-1]) + 1

        mid_mom     = mom[pos1:pos2]
        mid_mom_set = set(mid_mom)
        filter_dad  = [gene for gene in dad if gene not in mid_mom_set]
        gene_child_1 = filter_dad[:pos1] + mid_mom + filter_dad[pos1:]

        idx_best_dad           = int(np.argmin(fitness_sub_route_dad))
        best_fitness_sub_route_dad = float(fitness_sub_route_dad[idx_best_dad])
        if self.best_fitness_pD <= best_fitness_sub_route_dad or len(sub_route_dad) == 0:
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_dad = sub_route_dad[idx_best_dad]
            pos1 = dad.index(best_sub_route_dad[0])
            pos2 = dad.index(best_sub_route_dad[-1]) + 1

        mid_dad     = dad[pos1:pos2]
        mid_dad_set = set(mid_dad)
        filter_mom  = [gene for gene in mom if gene not in mid_dad_set]
        gene_child_2 = filter_mom[:pos1] + mid_dad + filter_mom[pos1:]

        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def PMX_crossover(self, dad, mom):
        size = len(mom)
        if size < 2:
            return dad, mom
        pos1, pos2 = sorted(random.sample(range(size), 2))

        gene_child_1 = [None] * size
        gene_child_2 = [None] * size

        mapping_gene         = {dad[i]: mom[i] for i in range(pos1, pos2)}
        reverse_mapping_gene = {mom[i]: dad[i] for i in range(pos1, pos2)}

        for idx_p in range(size):
            d = dad[idx_p]
            m = mom[idx_p]

            if pos1 <= idx_p < pos2:
                gene_child_1[idx_p] = mom[idx_p]
                gene_child_2[idx_p] = dad[idx_p]
                continue

            while d in reverse_mapping_gene:
                d = reverse_mapping_gene[d]

            while m in mapping_gene:
                m = mapping_gene[m]

            gene_child_1[idx_p] = d
            gene_child_2[idx_p] = m

        return gene_child_1, gene_child_2

    def best_cost_route_crossover(self, dad, mom):
        route_dad, _, _ = self.individual_to_route(dad)
        route_mom, _, _ = self.individual_to_route(mom)

        sub_route_mom = random.choice(route_mom)
        sub_route_dad = random.choice(route_dad)

        sub_route_mom_set = set(sub_route_mom)
        sub_route_dad_set = set(sub_route_dad)

        gene_child_1 = [gene for gene in dad if gene not in sub_route_mom_set]
        gene_child_2 = [gene for gene in mom if gene not in sub_route_dad_set]

        # Greedy search chạy trong numba (evaluator dùng chung)
        gene_child_1 = greedy_insert(
            np.array(gene_child_1, dtype=np.int64),
            np.array(sub_route_mom, dtype=np.int64),
            self.customers, self.graph_data, self._vehicle_capacity, self._M
        ).tolist()
        gene_child_2 = greedy_insert(
            np.array(gene_child_2, dtype=np.int64),
            np.array(sub_route_dad, dtype=np.int64),
            self.customers, self.graph_data, self._vehicle_capacity, self._M
        ).tolist()

        return gene_child_1, gene_child_2

    def STPB_crossover(self, dad, mom):
        # Đảm bảo hoạt động đúng dù nhận numpy array hay Python list
        dad = list(dad)
        mom = list(mom)
        probabilities = [0.25, 0.25, 0.25, 0.25]
        choice = np.random.choice(range(len(probabilities)), p=probabilities)

        match choice:
            case 0:
                gene_child_1, gene_child_2 = self.heuristic_single_point_crossover(dad, mom)
            case 1:
                gene_child_1, gene_child_2 = self.heuristic_two_point_crossover_v1(dad, mom)
            case 2:
                gene_child_1, gene_child_2 = self.PMX_crossover(dad, mom)
            case 3:
                gene_child_1, gene_child_2 = self.best_cost_route_crossover(dad, mom)

        return gene_child_1, gene_child_2

    def re_cluster_by_timewindow(self, clusters):
        """Gộp cụm theo time-window — delegate sang evaluator dùng chung."""
        return _re_cluster_fn(
            clusters, self.customers, self.graph_data,
            self._vehicle_capacity, self._M
        )

    def cal_number_of_clusters(self):
        customers = self.customers
        total_service  = np.sum(customers[:, C_SERVICE_TIME])
        check_capacity = np.sum(customers[:, C_DEMAND])
        check_due      = customers[0, C_DUE_TIME] - customers[0, C_READY_TIME] + self._M

        distance              = self.graph_data[1:]
        aver_dist             = np.mean(np.nonzero(distance))
        distance_to_depot     = self.graph_data[0, :]
        avg_distance_to_depot = np.mean(np.nonzero(distance_to_depot))

        total_moving_time = 2 * avg_distance_to_depot + (distance.shape[0] - 1) * aver_dist
        total_time        = total_service + total_moving_time

        return math.ceil(total_time / check_due), math.ceil(check_capacity / self._vehicle_capacity)

    def mutation(self, child):
        if len(child) < 4:
            return child
        pos1, pos2, pos3, pos4 = sorted(random.sample(range(len(child)), 4))
        return child[:pos1] + child[pos3:pos4 + 1] + child[pos2 + 1:pos3] + child[pos1:pos2 + 1] + child[pos4 + 1:]

    def hybird(self):
        index = math.floor(self._conserve_rate * self._individual)

        while len(self.population) < self._individual:
            hybird_rate = random.random()
            dad, mom = random.sample(self.population[index:], 2)

            if hybird_rate > self._crossover_rate:
                continue

            # .tolist() để crossover nhận Python list thuần — tránh vấn đề tương thích numpy
            gene_child_1, gene_child_2 = self.STPB_crossover(
                dad.customer_list.tolist(), mom.customer_list.tolist())

            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)

            # Chuyển về numpy khi tạo Individual
            self.population.append(Individual(customer_list=np.array(gene_child_1, dtype=np.int64)))

            if len(self.population) < self._individual:
                self.population.append(Individual(customer_list=np.array(gene_child_2, dtype=np.int64)))

    def fit(self, clusters):
        clusters = self.re_cluster_by_timewindow(clusters)
        for cluster in clusters:
            _start_time = time.time()
            self.initial_population(cluster)
            for _ in range(self._generation):
                self.cal_fitness_population()
                self.selection()
                self.hybird()

            self.process_time += round_float(time.time() - _start_time)
            self.cal_fitness_population()
            self.selection()
            self.best_fitness_global  += self.population[0].fitness
            self.best_distance_global += self.population[0].distance
            best_route, _, _ = self.individual_to_route(self.population[0].customer_list)
            self.best_route_global.append(best_route)
            self.route_count_global += len(best_route)
            self.best_fitness_pM = -1
            self.best_fitness_pD = -1

        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global, self.process_time


# Tất cả hàm numba (evaluate_route, analyze_route, evaluate_batch, greedy_insert)
# đã được chuyển vào ultility/vrptw_evaluator.py để dùng chung cho mọi thuật toán.