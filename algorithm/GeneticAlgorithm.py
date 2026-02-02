import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
import math
import random
from ultility.utilities import round_float, write_excel_file, distance_cdist, create_graph
import os
import time
import numba
from numba.core import types
from numba.typed import List

C_ID = 0
C_X = 1
C_Y = 2
C_DEMAND = 3
C_READY_TIME = 4
C_DUE_TIME = 5
C_SERVICE_TIME = 6

# Khách hàng
class Individual():
    def __init__(self, customer_list: list = None, fitness: float = 0, distance: float = 0):
        self.customer_list = customer_list
        self.fitness = fitness
        self.distance = distance

    def print(self):
        print(self.customer_list, ' ', self.fitness, ' ', self.distance)

# Thuật toán di truyền
class GA:
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.15, vehicle_capacity: float = 200, conserve_rate: float = 0.1, M: float = 0, customers: np.ndarray = None, graph_data: np.ndarray = None):
        self._individual = individual  # số cá thể
        self._generation = generation  # số thế hệ
        self._crossover_rate = crossover_rate  # tỉ lệ trao đổi chéo
        self._mutation_rate = mutation_rate  # tỉ lệ đột biến
        self._conserve_rate = conserve_rate  # tỉ lệ bảo tồn
        
        self.population = []

        self._vehicle_capacity = vehicle_capacity  # trọng tải của xe
        self._M = M  # sai số thời gian
        self.customers = customers  # Dữ liệu khách hàng
        self.graph_data = graph_data  # Dữ liệu đồ thị

        self.best_distance_global = 0
        self.route_count_global = 0
        self.best_fitness_global = 0
        self.best_route_global = []

        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

        self.process_time = 0

    # Tối ưu cục bộ(không kiểm tra điều kiện)
    def initial_population(self, cluster) -> list:
        self.population = [Individual(customer_list=random.sample(cluster, len(cluster))) for _ in range(self._individual)]
            

        # [a.print() for a in self.population]
        # print(isinstance(self.population[0].customer_list,np.ndarray))
        # exit()
        # print(self.population[-1].fitness)

    # Tách thành các lộ trình con
    @staticmethod
    @numba.njit(cache=True)
    def analyze_individual(individual: np.ndarray, customers: np.ndarray, graph_data: np.ndarray, vehicle_capacity: float, M: float):
        
        depot_due = customers[0, C_DUE_TIME] + M
        vehicle_load = 0  # trọng tải xe
        elapsed_time = 0  # Mốc thời gian hiện tại của xe
        last_customer_id = 0  # Vị khách đã xét trước đó
        split_indices = List.empty_list(types.int64)
        fitness_sub_routes = List.empty_list(types.float64)
        distance_sub_routes = List.empty_list(types.float64)
        fitness = 0
        distance = 0

        for i, customer_id in enumerate(individual):
            
            demand = customers[customer_id, C_DEMAND]
            ready_time = customers[customer_id, C_READY_TIME]
            due_time = customers[customer_id, C_DUE_TIME]
            service_time = customers[customer_id, C_SERVICE_TIME]

            # Thời gian di chuyển giữa 2 điểm
            moving_time = graph_data[customer_id, last_customer_id]
            # Mốc thời gian đến khách hàng thứ customer_id
            arrive_time = moving_time + elapsed_time
            # Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(ready_time - M - arrive_time, 0)
            delay_time = max(arrive_time - due_time - M, 0)
            # Thời gian rời khỏi điểm hiện tại (sau khi phục vụ và chờ)
            departure_time = arrive_time + waiting_time + service_time
            # Thời gian di chuyển từ điểm đang xét về kho
            return_time_from_current = graph_data[customer_id, 0]

            update_vehicle_load = vehicle_load + demand
            total_time_if_return = departure_time + return_time_from_current

            if (update_vehicle_load <= vehicle_capacity) and (total_time_if_return <= depot_due):
                vehicle_load = update_vehicle_load
                elapsed_time = departure_time
                distance += moving_time
                fitness += waiting_time + delay_time
            else:
                # Kết thúc sub_route hiện tại
                distance += graph_data[last_customer_id, 0]
                fitness += distance
                split_indices.append(i)
                distance_sub_routes.append(distance)
                fitness_sub_routes.append(fitness)

                # Bắt đầu lộ trình mới
                time_from_depot = graph_data[0, customer_id]

                # Cập nhật khoảng cách di chuyển từ điểm kết thúc về kho và bắt đầu từ kho đến điểm (Không cần phải tính phạt vì sẽ là bị dữ liệu sai)
                arrive_time_new_route = time_from_depot
                waiting_time = max(ready_time - M - arrive_time_new_route, 0)
                elapsed_time = arrive_time_new_route + waiting_time + service_time

                # Cập nhật lại fitness, distance, vehicle_load
                fitness = waiting_time
                distance = time_from_depot
                vehicle_load = demand

            last_customer_id = customer_id
        # Cập nhật lộ trình cuối cùng
        distance += graph_data[last_customer_id, 0]
        fitness += distance
        distance_sub_routes.append(distance)
        fitness_sub_routes.append(fitness)

        return split_indices, fitness_sub_routes, distance_sub_routes

    @staticmethod
    @numba.njit(cache=True)
    def cal_fitness_individualV2(individual: np.ndarray, customers: np.ndarray, graph_data: np.ndarray, vehicle_capacity: float, M: float, fitness_to_branch_bound: float=np.inf):
        """
        Hàm tính fitness và distance cho một cá thể, được tối ưu hóa cao bằng Numba.

        Args:
            individual (np.ndarray): Mảng 1D chứa thứ tự các khách hàng (ID).
            customer_data (np.ndarray): Mảng 2D chứa toàn bộ thông tin khách hàng.
            graph_data (np.ndarray): Ma trận khoảng cách/thời gian di chuyển.
            vehicle_capacity (float): Tải trọng tối đa của xe.
            M (float): Sai số thời gian cho phép.

        Returns:
            tuple[float, float]: Một tuple chứa (final_fitness, total_distance).
        """
        depot_due = customers[0, C_DUE_TIME] + M
        vehicle_load = 0  # trọng tải xe
        elapsed_time = 0  # Mốc thời gian hiện tại của xe
        last_customer_id = 0  # Vị khách đã xét trước đó
        fitness = 0
        distance = 0

        for customer_id in individual:
            demand = customers[customer_id, C_DEMAND]
            ready_time = customers[customer_id, C_READY_TIME]
            due_time = customers[customer_id, C_DUE_TIME]
            service_time = customers[customer_id, C_SERVICE_TIME]

            # Thời gian di chuyển giữa 2 điểm
            moving_time = graph_data[customer_id, last_customer_id]
            # Mốc thời gian đến khách hàng thứ customer_id
            arrive_time = moving_time + elapsed_time
            # Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(ready_time - M - arrive_time, 0)
            # Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
            delay_time = max(arrive_time - due_time - M, 0)
            # Thời gian rời khỏi điểm hiện tại (sau khi phục vụ và chờ)
            departure_time = arrive_time + waiting_time + service_time
            # Thời gian di chuyển từ điểm đang xét về kho
            return_time_from_current = graph_data[customer_id, 0]

            update_vehicle_load = vehicle_load + demand
            total_time_if_return = departure_time + return_time_from_current

            if (update_vehicle_load <= vehicle_capacity) and (total_time_if_return <= depot_due):
                vehicle_load = update_vehicle_load
                elapsed_time = departure_time
                distance += moving_time
                fitness += waiting_time + delay_time
            else:
                # Kết thúc sub_route hiện tại
                distance += graph_data[last_customer_id, 0]

                # Bắt đầu lộ trình mới
                time_from_depot = graph_data[0, customer_id]
                distance += time_from_depot

                # Cập nhật khoảng cách di chuyển từ điểm kết thúc về kho và bắt đầu từ kho đến điểm (Không cần phải tính phạt vì sẽ là bị dữ liệu sai)
                arrive_time_new_route = time_from_depot
                waiting_time = max(ready_time - M - arrive_time_new_route, 0)
                elapsed_time = arrive_time_new_route + waiting_time + service_time

                fitness += waiting_time
                vehicle_load = demand
            # print(fitness)
            last_customer_id = customer_id

            if fitness > fitness_to_branch_bound:
                return fitness, distance

        # Hoàn thành lộ trình cuối cùng bằng cách quay về kho
        distance += graph_data[last_customer_id, 0]
        fitness += distance

        return fitness, distance

    def individual_to_route(self, individual: np.ndarray):
        """
        Tách một cá thể (list) thành các lộ trình con (list của list).
        Hàm này sử dụng Numba để tăng tốc phần tính toán chính.
        """
        # Nếu individual rỗng, trả về list rỗng
        if not individual:
            return np.array([]),0,0

        # Chuyển list sang mảng NumPy để truyền vào hàm Numba
        individual_np = np.copy(individual)

        # 1. lấy các điểm ngắt
        split_points, fitness_sub_routes, distance_sub_routes = GA.analyze_individual(individual_np, self.customers, self.graph_data, self._vehicle_capacity, self._M)
            

        # 2. Dùng các điểm ngắt để tái tạo lại `route`
        route = []
        last_split_idx = 0
        for split_idx in split_points:
            # Lấy phần lộ trình từ điểm ngắt cuối cùng đến điểm ngắt hiện tại
            sub_route = individual[last_split_idx:split_idx]
            if sub_route:  # Chỉ thêm nếu sub_route không rỗng
                route.append(sub_route)
            last_split_idx = split_idx

        # Thêm phần lộ trình con cuối cùng (từ điểm ngắt cuối đến hết)
        final_sub_route = individual[last_split_idx:]
        if final_sub_route:
            route.append(final_sub_route)

        return route, fitness_sub_routes, distance_sub_routes

    # Tính mức độ thích nghi trên toàn bộ quần thể
    def cal_fitness_population(self):
        for individual in self.population:
            individual.fitness, individual.distance = GA.cal_fitness_individualV2(
                individual=np.array(individual.customer_list, dtype=np.int64), customers=self.customers, graph_data=self.graph_data, vehicle_capacity=self._vehicle_capacity, M=self._M)

        # [a.print() for a in self.population]
        # exit()
        # print(self.population)

    def selection(self):
        # Sắp xếp quần thể theo chiều tăng dần
        self.population.sort(key=lambda x: x.fitness)
        # [a.print() for a in self.population]
        # vị trí xóa = (1-tỉ lệ bảo tồn)*số cá thể/2 + tỉ lệ bảo tồn *số cá thể
        positionToDel = math.floor(self._individual*(1+self._conserve_rate)/2)
        del self.population[positionToDel:]

    def single_point_crossover(self, dad, mom):
        assert len(dad) == len(mom), "Dad and Mom must have the same length."

        pos1 = random.randrange(len(mom))

        # Lấy phần tử ở dad không xuất hiện trong mom[pos1:]
        mom_tail_set = set(mom[pos1:])
        filter_dad = [gene for gene in dad if gene not in mom_tail_set]

        # Lấy phần tử ở mom không xuất hiện trong dad[pos1:]
        dad_tail_set = set(dad[pos1:])
        filter_mom = [gene for gene in mom if gene not in dad_tail_set]

        # Ghép gene
        gene_child_1 = filter_dad[:pos1] + mom[pos1:]
        gene_child_2 = filter_mom[:pos1] + dad[pos1:]

        return gene_child_1, gene_child_2

    def heuristic_single_point_crossover(self, dad, mom):
        sub_route_mom, fitness_sub_route_mom, _ = self.individual_to_route(mom)
        sub_route_dad, fitness_sub_route_dad, _ = self.individual_to_route(dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if self.best_fitness_pM <= best_fitness_sub_route_mom:
            pos1 = random.randrange(len(mom))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(
                best_fitness_sub_route_mom)]
            # tìm vị trí phần tử trong list
            pos1 = mom.index(best_sub_route_mom[0])

        # Tạo con 1
        mom_tail_set = set(mom[pos1:])
        filter_dad = [gene for gene in dad if gene not in mom_tail_set]
        gene_child_1 = filter_dad[:pos1] + mom[pos1:]

        # ==========================================

        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if self.best_fitness_pD <= best_fitness_sub_route_dad:
            pos1 = random.randrange(len(dad))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(
                best_fitness_sub_route_dad)]
            pos1 = dad.index(best_sub_route_dad[0])

        # Tạo con 2
        dad_tail_set = set(dad[pos1:])
        filter_mom = [gene for gene in mom if gene not in dad_tail_set]
        gene_child_2 = filter_mom[:pos1] + dad[pos1:]

        # Cập nhật best fitness
        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def two_point_crossover(self, dad, mom):
        # Chuyển đổi cá thể thành các sub_route
        sub_route_mom, _, _ = self.individual_to_route(mom)
        sub_route_dad, _, _ = self.individual_to_route(dad)

        # Chọn điểm cắt cho mẹ
        pos1, pos2 = sorted(random.choices(range(len(sub_route_mom)), k=2))

        sub_mom = [item for sublist in sub_route_mom[pos1:pos2+1]
                   for item in sublist]

        pos1, pos2 = sorted(random.choices(range(len(sub_route_dad)), k=2))

        sub_dad = [item for sublist in sub_route_dad[pos1:pos2+1]
                   for item in sublist]

        # Tạo filter_dad bằng cách loại bỏ phần tử của sub_mom
        filter_dad = [item for item in dad if item not in sub_mom]
        gene_child_1 = filter_dad[:pos1] + sub_mom + filter_dad[pos1:]

        # Tạo filter_mom bằng cách loại bỏ phần tử của sub_dad
        filter_mom = [item for item in mom if item not in sub_dad]
        gene_child_2 = filter_mom[:pos1] + sub_dad + filter_mom[pos1:]

        return gene_child_1, gene_child_2

    def heuristic_two_point_crossover_v1(self, dad, mom):
        size = len(mom)

        sub_route_mom, fitness_sub_route_mom, _ = self.individual_to_route(mom)
        sub_route_dad, fitness_sub_route_dad, _ = self.individual_to_route(dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if self.best_fitness_pM <= best_fitness_sub_route_mom:
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(
                best_fitness_sub_route_mom)]
            pos1 = mom.index(best_sub_route_mom[0])
            # chú ý +1 vì slicing bên phải mở (khác np)
            pos2 = mom.index(best_sub_route_mom[-1]) + 1

        # Tạo gene_child_1
        mid_mom = mom[pos1:pos2]
        mid_mom_set = set(mid_mom)
        filter_dad = [gene for gene in dad if gene not in mid_mom_set]
        gene_child_1 = filter_dad[:pos1] + mid_mom + filter_dad[pos1:]

        # ========================

        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if self.best_fitness_pD <= best_fitness_sub_route_dad:
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(
                best_fitness_sub_route_dad)]
            pos1 = dad.index(best_sub_route_dad[0])
            pos2 = dad.index(best_sub_route_dad[-1]) + 1  # chú ý +1

        # Tạo gene_child_2
        mid_dad = dad[pos1:pos2]
        mid_dad_set = set(mid_dad)
        filter_mom = [gene for gene in mom if gene not in mid_dad_set]
        gene_child_2 = filter_mom[:pos1] + mid_dad + filter_mom[pos1:]

        # Cập nhật best fitness
        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def PMX_crossover(self, dad, mom):
        size = len(mom)
        pos1, pos2 = sorted(random.sample(range(size), 2))

        # Tạo gene con, ban đầu toàn None
        gene_child_1 = [None] * size
        gene_child_2 = [None] * size

        # Tạo ánh xạ
        mapping_gene = {dad[i]: mom[i] for i in range(pos1, pos2)}
        reverse_mapping_gene = {mom[i]: dad[i] for i in range(pos1, pos2)}

        # Ánh xạ cho mỗi vị trí
        for idx_p in range(size):
            d = dad[idx_p]
            m = mom[idx_p]

            if pos1 <= idx_p < pos2:
                gene_child_1[idx_p] = mom[idx_p]
                gene_child_2[idx_p] = dad[idx_p]
                continue

            # Ánh xạ phần gen của bố
            while d in reverse_mapping_gene:
                d = reverse_mapping_gene[d]

            # Ánh xạ phần gen của mẹ
            while m in mapping_gene:
                m = mapping_gene[m]

            gene_child_1[idx_p] = d
            gene_child_2[idx_p] = m

        return gene_child_1, gene_child_2

    def best_cost_route_crossover(self, dad, mom):
        # Tách các route con từ các lộ trình cha và mẹ
        route_dad, _, _ = self.individual_to_route(dad)
        route_mom, _, _ = self.individual_to_route(mom)

        # Chọn ngẫu nhiên sub_route từ cha và mẹ
        sub_route_mom = random.choice(route_mom)
        sub_route_dad = random.choice(route_dad)

        # Tráo đổi cho nhau và tiến hành xóa những phần tử đấy
        sub_route_mom_set = set(sub_route_mom)
        sub_route_dad_set = set(sub_route_dad)

        gene_child_1 = [gene for gene in dad if gene not in sub_route_mom_set]
        gene_child_2 = [gene for gene in mom if gene not in sub_route_dad_set]

        # Sau đó lấp lại bằng cách dùng phương pháp tham lam

        def greedySearch(intersect_gene, diff_gene):
            common_part = list(intersect_gene)
            # Thực hiện thuật toán nhánh cận
            # diff_gene: đoạn gen lắp vào
            # common_part: đoạn gen trả về
            for gen in diff_gene:
                idx_cus_min = 0
                fitness_min = float('inf')
                # Tìm vị trí chèn tốt nhất cho gen này
                for idx in range(len(common_part) + 1):
                    fitness_part, _ = GA.cal_fitness_individualV2(individual=np.array(common_part[:idx] + [gen] + common_part[idx:], dtype=np.int64), customers=self.customers,
                                                                  graph_data=self.graph_data, vehicle_capacity=self._vehicle_capacity, M=self._M, fitness_to_branch_bound=fitness_min)

                    if fitness_min > fitness_part:
                        fitness_min = fitness_part
                        idx_cus_min = idx

                # Chèn khách hàng
                common_part.insert(idx_cus_min, gen)
            return common_part

        gene_child_1 = greedySearch(gene_child_1, sub_route_mom)
        gene_child_2 = greedySearch(gene_child_2, sub_route_dad)

        return gene_child_1, gene_child_2

    def STPB_crossover(self, dad, mom):  # 3411.464
        probabilities = [0.25, 0.25, 0.25, 0.25]

        choice = np.random.choice(range(len(probabilities)), p=probabilities)  

        match choice:
            case 0:
                gene_child_1, gene_child_2 = self.heuristic_single_point_crossover(
                    dad, mom)
            case 1:
                gene_child_1, gene_child_2 = self.heuristic_two_point_crossover_v1(
                    dad, mom)  # Order
            case 2:
                gene_child_1, gene_child_2 = self.PMX_crossover(dad, mom)
            case 3:
                gene_child_1, gene_child_2 = self.best_cost_route_crossover(
                    dad, mom)

        return gene_child_1, gene_child_2

    def re_cluster_by_timewindow(self, clusters):

        def check_concatenate(cluster1, cluster2):
            customers = self.customers
            
            # Nối hai cụm
            total_cluster = cluster1 + cluster2

            # Tính thời gian phục vụ cho toàn bộ khách hàng
            total_service = np.sum(customers[total_cluster, C_SERVICE_TIME])

            # Tính trọng tải của toàn bộ khách hàng
            check_capacity = np.sum(customers[total_cluster, C_DEMAND])
            
            # Tính thời gian xe hoạt động tối đa
            check_due = customers[0, C_DUE_TIME] - customers[0, C_READY_TIME] + self._M

            # Kiểm tra điều kiện ràng buộc trọng tải
            if check_capacity > self._vehicle_capacity:
                return False

            # Tính khoảng cách giữa các điểm trong total_cluster_indices
            distance = self.graph_data[np.ix_(total_cluster, total_cluster)]

            # Tính khoảng cách trung bình giữa 2 điểm khách hàng trong cụm ghép
            aver_dist = np.mean(np.nonzero(distance))

            # Tính khoảng cách từ depot đến các điểm trong cụm ghép
            distance_to_depot = self.graph_data[0, total_cluster]

            # Khoảng cách trung bình từ kho đến cụm ghép
            avg_distance_to_depot = np.mean(np.nonzero(distance_to_depot))
            
            # Thời gian di chuyển ước lượng = 2 * khoảng cách trung bình từ kho đến cụm + (số lượng khách hàng -1) * khoảng cách trung bình giữa 2 điểm + thời gian phục vụ toàn bộ khách hàng
            # số lượng khách hàng - 1 là số lượng cạch
            total_moving_time = 2 * avg_distance_to_depot + (len(total_cluster) - 1) * aver_dist
            
            # Tổng thời gian phục vụ và di chuyển
            total_time = total_service + total_moving_time

            return total_time <= check_due

        def concatenate_arrays(array, index1, index2):
            # Nối hai mảng theo chỉ số đã cho
            return [array[index1] + array[index2]] + [array[i] for i in range(len(array)) if i != index1 and i != index2]

        i = 0
        while i < len(clusters) - 1:
            j = i + 1
            while j < len(clusters):
                if check_concatenate(clusters[i], clusters[j]):
                    clusters = concatenate_arrays(clusters, i, j)
                    # Sau khi nối, không cần kiểm tra lại j
                    j = i + 1  # Reset j để kiểm tra lại từ i
                else:
                    j += 1  # Chỉ tăng j nếu không nối
            i += 1

        return clusters

    def cal_number_of_clusters(self):
        customers = self.customers
        # Tính thời gian phục vụ cho toàn bộ khách hàng
        total_service = np.sum(customers[:, C_SERVICE_TIME])

        # Tính trọng tải của toàn bộ khách hàng
        check_capacity = np.sum(customers[:, C_DEMAND])
        
        # Tính thời gian xe hoạt động tối đa
        check_due = customers[0, C_DUE_TIME] - customers[0, C_READY_TIME] + self._M
        
        # Tính khoảng cách giữa các điểm trong total_cluster_indices
        distance = self.graph_data[1:]

        # Tính khoảng cách trung bình giữa 2 điểm khách hàng trong cụm ghép
        aver_dist = np.mean(np.nonzero(distance))

        # Tính khoảng cách từ depot đến các điểm trong cụm ghép
        distance_to_depot = self.graph_data[0, :]

        # Khoảng cách trung bình từ kho đến cụm ghép
        avg_distance_to_depot = np.mean(np.nonzero(distance_to_depot))
        
         # Thời gian di chuyển ước lượng = 2 * khoảng cách trung bình từ kho đến cụm + (số lượng khách hàng -1) * khoảng cách trung bình giữa 2 điểm + thời gian phục vụ toàn bộ khách hàng
        # số lượng khách hàng - 1 là số lượng cạch
        total_moving_time = 2 * avg_distance_to_depot + (distance.shape[0] - 1) * aver_dist
            
            # Tổng thời gian phục vụ và di chuyển
        total_time = total_service + total_moving_time
        
        
        return math.ceil(total_time / check_due), math.ceil(check_capacity / self._vehicle_capacity)

    def mutation(self, child):
        child_new = child.copy()  # Tạo bản sao của child (dùng list.copy() thay vì np.copy)

        pos1, pos2, pos3, pos4 = sorted(random.sample(range(len(child)), 4))

        # Thực hiện thao tác hoán vị các phần tử trong child
        child_new = child[:pos1] + child[pos3:pos4+1] + \
            child[pos2+1:pos3] + child[pos1:pos2+1] + child[pos4+1:]

        return child_new

    def hybird(self):
        # Lấy vị trí bắt đầu của cá thể được phép lai ghép trong quần thể
        index = math.floor(self._conserve_rate*self._individual)

        while (len(self.population) < self._individual):
            # Lấy ra tỉ lệ sinh của quần thể lần này
            hybird_rate = random.random()

            # Lấy ngẫu nhiên ra 2 cá thể đem trao đổi chéo
            dad, mom = random.sample(self.population[index:], 2)

            # Kiểm tra tỉ lệ sinh so với tủi lệ trao đổi chéo
            if (hybird_rate > self._crossover_rate):
                continue

            # Tiến hành trao đổi chéo
            gene_child_1, gene_child_2 = self.STPB_crossover(
                dad.customer_list, mom.customer_list)

            # Kiếm tra tỉ lệ sinh với tỉ lệ đột biến gen
            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)

            child1 = Individual(customer_list=gene_child_1)
            self.population.append(child1)

            if (len(self.population) < self._individual):
                child2 = Individual(customer_list=gene_child_2)
                self.population.append(child2)

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
            self.best_fitness_global += self.population[0].fitness
            self.best_distance_global += self.population[0].distance
            best_route, _, _ = self.individual_to_route(self.population[0].customer_list)
            self.best_route_global.append(best_route)
            self.route_count_global += len(best_route)
            self.best_fitness_pM = -1
            self.best_fitness_pD = -1
            
        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global, self.process_time

