import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
import math
import random
from ultility.utilities import round_float, write_excel_file, distance_cdist
import os
import time

# Khách hàng


class Individual():
    def __init__(self, customerList: np.array = None, fitness: float = 0, distance: float = 0):
        self.customerList = customerList
        self.fitness = fitness
        self.distance = distance

    def print(self):
        print(self.customerList, ' ', self.fitness, ' ', self.distance)

# Thuật toán di truyền


class GA:
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.15, vehcicle_capacity: float = 200, conserve_rate: float = 0.1, M: float = 50, customers: list = None):
        self._individual = individual  # số cá thể
        self._generation = generation  # số thế hệ
        self._crossover_rate = crossover_rate  # tỉ lệ trao đổi chéo
        self._mutation_rate = mutation_rate  # tỉ lệ đột biến
        self._vehcicle_capacity = vehcicle_capacity  # trọng tải của xe
        self._conserve_rate = conserve_rate  # tỉ lệ bảo tồn
        self._M = M  # sai số thời gian
        self.customers = customers  # Dữ liệu khách hàng

        self.best_distance_global = 0
        self.route_count_global = 0
        self.best_fitness_global = 0
        self.best_route_global = []

        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

        self.process_time = 0


    # Tối ưu cục bộ(không kiểm tra điều kiện)
    def initialPopulation(self, cluster) -> np.ndarray:
        self.__population = [Individual(customerList=np.random.permutation(
            cluster)) for _ in range(self._individual)]
        # [a.print() for a in self.__population]
        # print(isinstance(self.__population[0].customerList,np.ndarray))
        # exit()
        # print(self.__population[-1].fitness)

    # Tách thành các lộ trình con
    def individualToRoute(self, individual):
        route = []  # Lộ trình tổng
        vehicle_load = 0  # trọng tải xe
        sub_route = []  # Lộ trình con
        elapsed_time = 0  # Mốc thời gian hiện tại của xe
        last_customer_id = 0  # Vị khách đã xét trước đó
        for customer_id in individual:
            customer = self.customers[customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand

            # Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            moving_time = np.linalg.norm(
                customer.xy_coord-self.customers[last_customer_id].xy_coord)

            # Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M -
                               elapsed_time - moving_time, 0)
            # Thời gian di chuyển từ điểm đang xét về kho
            return_time = np.linalg.norm(
                customer.xy_coord-self.customers[0].xy_coord)

            update_elapsed_time = elapsed_time + service_time + \
                moving_time + waiting_time + return_time

            if (update_vehicle_load <= self._vehcicle_capacity) and (update_elapsed_time <= self.customers[0].dueTime + self._M):
                sub_route.append(customer_id)
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
                elapsed_time = service_time + return_time + \
                    max(customer.readyTime - self._M - return_time, 0)
            last_customer_id = customer_id
        if sub_route:
            # Lưu lộ trình con còn lại sau khi xét hết các điểm
            route.append(sub_route)
        route = [arr for arr in route if len(arr) > 0]

        return route  # 1 list các array

    # Tính mức độ thích nghi trên toàn bộ quần thể
    def cal_fitness_population(self):
        for individual in self.__population:
            individual.fitness, individual.distance = self.cal_fitness_individualV2(
                individual.customerList)
        # [a.print() for a in self.__population]
        # exit()
        # print(self.__population)

    def cal_fitness_individual(self, individual):
        route = self.individualToRoute(individual)
        fitness = 0
        distance = 0
        for sub_route in route:
            sub_route_time_cost = 0  # Thời gian đợi và phạt của 1 route con
            sub_route_distance = 0  # Thời gian di chuyển của 1 route con
            elapsed_time = 0
            last_customer_id = 0
            for customer_id in sub_route:
                customer = self.customers[customer_id]

                # Thời gian di chuyển giữa 2 điểm
                moving_time = np.linalg.norm(
                    customer.xy_coord-self.customers[last_customer_id].xy_coord)

                # Cập nhật thời gian di chuyển
                sub_route_distance += moving_time

                # Mốc thời gian đến khách hàng thứ customer_id
                arrive_time = moving_time + elapsed_time

                # Thời gian chờ đợi nếu xe đang ở thời gian chưa đến thời gian bắt đầu phục vụ
                waiting_time = max(customer.readyTime -
                                   self._M - arrive_time, 0)

                # Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
                delay_time = max(arrive_time - customer.dueTime - self._M, 0)

                # Cập nhật thời gian đợi và phạt
                sub_route_time_cost += waiting_time + delay_time
                # Cập nhật mốc thời gian mới của xe
                elapsed_time = arrive_time + customer.serviceTime + waiting_time
                # Cập nhật khách hàng đang xét
                last_customer_id = customer_id
            # Thời gian di chuyển từ điểm cuối cùng về kho
            return_time = np.linalg.norm(
                self.customers[last_customer_id].xy_coord-self.customers[0].xy_coord)
            sub_route_distance += return_time
            fitness += sub_route_distance + sub_route_time_cost
            distance += sub_route_distance

        return fitness, distance

    def cal_fitness_individualV2(self, individual):
        vehicle_load = 0  # trọng tải xe
        elapsed_time = 0  # Mốc thời gian hiện tại của xe
        last_customer_id = 0  # Vị khách đã xét trước đó
        sub_route = []  # Lộ trình con
        fitness = 0
        distance = 0
        depot = self.customers[0]  # Lấy dữ liệu kho
        for customer_id in individual:
            customer = self.customers[customer_id]
            last_customer = self.customers[last_customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand

            # Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            moving_time = np.linalg.norm(
                customer.xy_coord-last_customer.xy_coord)

            # Mốc thời gian đến khách hàng thứ customer_id
            arrive_time = moving_time + elapsed_time

            # Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M - arrive_time, 0)

            # Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
            delay_time = max(arrive_time - customer.dueTime - self._M, 0)

            # Thời gian di chuyển từ điểm đang xét về kho
            return_time = np.linalg.norm(customer.xy_coord-depot.xy_coord)

            update_elapsed_time = arrive_time + service_time + waiting_time + return_time

            if (update_vehicle_load <= self._vehcicle_capacity) and (update_elapsed_time <= depot.dueTime + self._M):
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
                sub_route.append(customer_id)
                # Cập nhật thời gian di chuyển
                distance += moving_time
                # Cập nhật thời gian đợi và phạt
                fitness += waiting_time + delay_time
            else:
                distance += return_time + \
                    np.linalg.norm(last_customer.xy_coord-depot.xy_coord)
                sub_route = [customer_id]
                # Cập nhật khoảng cách di chuyển từ điểm kết thúc về kho và bắt đầu từ kho đến điểm (Không cần phải tính phạt vì sẽ là bị dữ liệu sai)
                waiting_time = max(customer.readyTime -
                                   self._M - return_time, 0)
                fitness += waiting_time
                vehicle_load = demand
                elapsed_time = service_time + return_time + waiting_time

            # print(fitness)
            last_customer_id = customer_id

        if sub_route != []:
            distance += np.linalg.norm(
                self.customers[last_customer_id].xy_coord-depot.xy_coord)
            fitness += distance

        return fitness, distance

    def cal_fitness_sub_route(self, route):
        sub_route_result = []
        for sub_route in route:
            fitness, distance = self.cal_fitness_individualV2(sub_route)
            sub_route_result.append(fitness)
        return sub_route_result

    def selection(self):
        # Sắp xếp quần thể theo chiều tăng dần
        self.__population.sort(key=lambda x: x.fitness)
        # [a.print() for a in self.__population]
        # vị trí xóa = (1-tỉ lệ bảo tồn)*số cá thể/2 + tỉ lệ bảo tồn *số cá thể
        positionToDel = math.floor(self._individual*(1+self._conserve_rate)/2)
        del self.__population[positionToDel:]

    def SinglePointCrossover(self, dad, mom):
        # Chọn 1 điểm cắt ngẫu nhiên
        pos1 = random.randrange(len(mom))

        filter_dad = np.setdiff1d(dad, mom[pos1:], assume_unique=True)
        filter_mom = np.setdiff1d(mom, dad[pos1:], assume_unique=True)

        gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:]))
        gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:]))

        # print('gene_child_1',np.sort(gene_child_1))
        # print('gene_child_2',np.sort(gene_child_2))

        return gene_child_1, gene_child_2

    def heuristic_SinglePointCrossover(self, dad, mom):  # 3700.171
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        fitness_sub_route_mom = self.cal_fitness_sub_route(sub_route_mom)
        fitness_sub_route_dad = self.cal_fitness_sub_route(sub_route_dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if (self.best_fitness_pM <= best_fitness_sub_route_mom):
            # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
            pos1 = random.randrange(len(mom))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(
                best_fitness_sub_route_mom)]
            # pos1 = mom.index(best_sub_route_mom[0])
            pos1 = np.argwhere(mom == best_sub_route_mom[0])[0][0]

        filter_dad = np.setdiff1d(dad, mom[pos1:], assume_unique=True)
        gene_child_1 = np.hstack((filter_dad[:pos1], mom[pos1:]))
        # #==========================================
        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if (self.best_fitness_pD <= best_fitness_sub_route_dad):
            # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
            pos1 = random.randrange(len(dad))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(
                best_fitness_sub_route_dad)]
            # pos1 = dad.index(best_sub_route_dad[0])
            pos1 = np.argwhere(dad == best_sub_route_dad[0])[0][0]

        filter_mom = np.setdiff1d(mom, dad[pos1:], assume_unique=True)
        gene_child_2 = np.hstack((filter_mom[:pos1], dad[pos1:]))

        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def heuristic_SinglePointCrossoverV2(self, dad, mom):  # 4162.935
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        # Lấy ra 1 khách hàng ngẫu nhiên
        customer = mom[random.randrange(len(mom))]

        for sub_mom in sub_route_mom:
            if (customer in sub_mom):
                sub_route_mom = mom[np.argwhere(mom == sub_mom[0])[0][0]:]
                break

        for sub_dad in sub_route_dad:
            if (customer in sub_dad):
                sub_route_dad = dad[np.argwhere(dad == sub_dad[0])[0][0]:]
                break

        union_gene = np.union1d(sub_route_dad, sub_route_mom)
        intersect_gene = np.intersect1d(
            sub_route_dad, sub_route_mom, assume_unique=True)  # Đoạn gen chung
        # Đoạn gen cần sắp xêp và cho vào
        diff_gene = np.setdiff1d(union_gene, intersect_gene)

        def greedySearch(intersect_gene, diff_gene):
            commom_part = np.copy(intersect_gene)
            # Thực hiện thuật toán tham lam
            for gen in diff_gene:
                customer = customers[gen]
                min = 1.e10
                idx_cus_min = -1  # Vị trí khách hàng có fitness thấp nhất khi di chuyển từ i đến gen
                for idx, i in enumerate(commom_part):
                    moving_time = np.linalg.norm(
                        customers[i].xy_coord-customer.xy_coord)
                    waiting_time = max(customer.readyTime -
                                       self._M - moving_time, 0)
                    delay_time = max(
                        moving_time - customer.dueTime - self._M, 0)
                    if (min > waiting_time+delay_time):
                        min = waiting_time+delay_time
                        idx_cus_min = idx
                # Chèn khách hàng gen ở ngay sau cus_min
                commom_part = np.insert(commom_part, idx_cus_min+1, gen)
            return commom_part

       # Đưa ra các chỉ số mà mảng dad và mom có trong union_gene
        mom_idx = np.intersect1d(
            mom, union_gene, assume_unique=True, return_indices=True)[1]

        gene_child_1 = np.copy(mom)
        commom_part = greedySearch(intersect_gene, diff_gene)
        for idx, val in enumerate(mom_idx):
            gene_child_1[val] = commom_part[idx]

        dad_idx = np.intersect1d(
            dad, union_gene, assume_unique=True, return_indices=True)[1]
        gene_child_2 = np.copy(dad)
        commom_part = greedySearch(intersect_gene, np.flip(diff_gene))
        for idx, val in enumerate(dad_idx):
            gene_child_2[val] = commom_part[idx]

        return gene_child_1, gene_child_2

    def TwoPointCrossover(self, dad, mom):  # Order
        size = len(mom)  # Độ dài chuỗi gen
        # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
        pos1, pos2 = sorted(random.sample(range(size), 2))

        # Tìm ra các phần tử chưa được sử dụng ở cặp cha mẹ
        # filter_dad = [x for x in dad if x not in gene_child_1]
        filter_dad = np.setdiff1d(dad, mom[pos1:pos2], assume_unique=True)
        # filter_mom = [x for x in mom if x not in gene_child_2]
        filter_mom = np.setdiff1d(mom, dad[pos1:pos2], assume_unique=True)

        gene_child_1 = np.hstack(
            (filter_dad[:pos1], mom[pos1:pos2], filter_dad[pos1:]))
        gene_child_2 = np.hstack(
            (filter_mom[:pos1], dad[pos1:pos2], filter_mom[pos1:]))

        return gene_child_1, gene_child_2

    # ORDER cải tiến không theo hướng heuristic 3433.901
    def TwoPointOrderCrossover(self, dad, mom):
        size = len(mom)  # Độ dài chuỗi gen
        # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
        pos1, pos2 = np.sort(random.sample(range(size), 2))

        # Kiểm tra chuỗi gen của mẹ được cắt có tồn tại trong gen bố
        pos1_dad, pos2_dad = np.sort(
            [np.argwhere(dad == mom[pos1])[0][0], np.argwhere(dad == mom[pos2])[0][0]])
        pos1_mom, pos2_mom = np.sort(
            [np.argwhere(mom == dad[pos1])[0][0], np.argwhere(mom == dad[pos2])[0][0]])

        if (len(mom[pos1:pos2]) == len(dad[pos1_dad:pos2_dad]) and all(mom[pos1:pos2] == dad[pos1_dad:pos2_dad])):
            pos2_dad += random.randint(0, size-len(dad[pos1:pos2]))

        if (len(dad[pos1:pos2]) == len(mom[pos1_mom:pos2_mom]) and all(dad[pos1:pos2] == mom[pos1_mom:pos2_mom])):
            pos2_mom += random.randint(0, size-len(mom[pos1:pos2]))

        filter_dad = np.array([dad[(x+pos2_dad) % size] for x in range(size)
                              if dad[(x+pos2_dad) % size] not in mom[pos1:pos2]])
        filter_mom = np.array([mom[(x+pos2_mom) % size] for x in range(size)
                              if mom[(x+pos2_mom) % size] not in dad[pos1:pos2]])

        # gene_child_1[pos2:] = filter_dad[:len(gene_child_1[pos2:])]
        # gene_child_1[:pos1] = filter_dad[len(gene_child_1[pos2:]):]
        gene_child_1 = np.hstack(
            (filter_dad[len(mom[pos2:]):], mom[pos1:pos2], filter_dad[:len(mom[pos2:])]))
        # gene_child_2[pos2:] = filter_mom[:len(gene_child_2[pos2:])]
        # gene_child_2[:pos1] = filter_mom[len(gene_child_2[pos2:]):]
        gene_child_2 = np.hstack(
            (filter_mom[len(dad[pos2:]):], dad[pos1:pos2], filter_mom[:len(dad[pos2:])]))

        return gene_child_1, gene_child_2

    # 3452.482 Order Crossover
    def heuristic_TwoPointCrossoverV1(self, dad, mom):
        size = len(mom)  # Độ dài chuỗi gen

        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        fitness_sub_route_mom = self.cal_fitness_sub_route(sub_route_mom)
        fitness_sub_route_dad = self.cal_fitness_sub_route(sub_route_dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if (self.best_fitness_pM <= best_fitness_sub_route_mom):
            # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(
                best_fitness_sub_route_mom)]
            # pos1 = mom.index(best_sub_route_mom[0])
            pos1 = np.argwhere(mom == best_sub_route_mom[0])[0][0]
            # pos2 = mom.index(best_sub_route_mom[-1])
            pos2 = np.argwhere(mom == best_sub_route_mom[-1])[0][0]

        filter_dad = np.setdiff1d(dad, mom[pos1:pos2], assume_unique=True)
        gene_child_1 = np.hstack(
            (filter_dad[:pos1], mom[pos1:pos2], filter_dad[pos1:]))

        # ===============
        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if (self.best_fitness_pD <= best_fitness_sub_route_dad):
            # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(
                best_fitness_sub_route_dad)]
            pos1 = np.argwhere(dad == best_sub_route_dad[0])[0][0]
            pos2 = np.argwhere(dad == best_sub_route_dad[-1])[0][0]

        filter_mom = np.setdiff1d(mom, dad[pos1:pos2], assume_unique=True)
        gene_child_2 = np.hstack(
            (filter_mom[:pos1], dad[pos1:pos2], filter_mom[pos1:]))

        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def heuristic_TwoPointCrossoverV2(self, dad, mom):  # Order Greedy 4013.508
        # Lấy ra từng lộ trình con
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)
        # Tìm theo giá trị llen dài nhất

        # # sub_route = np.array([len(sub) for sub in sub_route_mom])
        # sub_route = self.cal_fitness_sub_route(sub_route_mom)

        # sub_route_mom = sub_route_mom[np.argmin(sub_route)]

        # Lấy ra 1 khách hàng ngẫu nhiên
        customer = mom[random.randrange(len(mom))]
        # customer = np.random.choice(sub_route_mom)

        # print("customer",customer)

        # Lấy ra lộ trình con có trong cha và mẹ
        for sub_mom in sub_route_mom:
            if (customer in sub_mom):
                sub_route_mom = sub_mom
                break
        for sub_dad in sub_route_dad:
            if (customer in sub_dad):
                sub_route_dad = sub_dad
                break

        union_gene = np.union1d(sub_route_dad, sub_route_mom)
        intersect_gene = np.intersect1d(
            sub_route_dad, sub_route_mom, assume_unique=True)  # Đoạn gen chung
        # Đoạn gen cần sắp xêp và cho vào
        diff_gene = np.setdiff1d(union_gene, intersect_gene)

        def greedySearch(intersect_gene, diff_gene):
            commom_part = np.copy(intersect_gene)
            # Thực hiện thuật toán tham lam
            # diff_gene: đoạn gen lắp vào
            # common_part: đoạn gen trả về
            for gen in diff_gene:

                parts = [self.cal_fitness_individualV2(np.insert(commom_part, idx, gen))[
                    0] for idx in range(len(commom_part)+1)]
                idx_cus_min = np.argmin(parts)

                # Chèn khách hàng
                commom_part = np.insert(commom_part, idx_cus_min, gen)
            return commom_part

        # Đưa ra các chỉ số mà mảng dad và mom có trong union_gene
        mom_idx = np.intersect1d(
            mom, union_gene, assume_unique=True, return_indices=True)[1]

        gene_child_1 = np.copy(mom)
        commom_part = greedySearch(intersect_gene, diff_gene)
        for idx, val in enumerate(mom_idx):
            gene_child_1[val] = commom_part[idx]

        dad_idx = np.intersect1d(
            dad, union_gene, assume_unique=True, return_indices=True)[1]
        gene_child_2 = np.copy(dad)
        commom_part = greedySearch(intersect_gene, np.flip(diff_gene))
        for idx, val in enumerate(dad_idx):
            gene_child_2[val] = commom_part[idx]

        return gene_child_1, gene_child_2

    def OPEXCrossover(self, dad, mom):  # 3783.941
        size = len(mom)  # Độ dài chuỗi gen
        pos1 = random.randrange(len(mom))

        gene_child_1 = np.zeros(len(mom), dtype=int)
        gene_child_2 = np.zeros(len(dad), dtype=int)

        # Lắp phần gen của bố mẹ cho 2 con
        gene_child_1[:pos1] = np.copy(mom[:pos1])
        gene_child_2[:pos1] = np.copy(dad[:pos1])

        # Lấy đoạn gen được cắt ra
        cut_gen_mom = np.copy(mom[pos1:])
        cut_gen_dad = np.copy(dad[pos1:])

        # # Sắp xếp đoạn gen theo tiêu chí
        # sort_cut_mom = self.sort_cluster_by_distance(cut_gen_mom)
        # sort_cut_dad = self.sort_cluster_by_distance(cut_gen_dad)

        # # Chuyển thành chỉ số sắp xếp (tương tự argsort)
        # for i in range(len(cut_gen_dad)):
        #     sort_cut_dad[i] = np.argwhere(cut_gen_dad == sort_cut_dad[i])[0][0]
        #     sort_cut_mom[i] = np.argwhere(cut_gen_mom == sort_cut_mom[i])[0][0]
        sort_cut_mom = np.argsort(cut_gen_mom)
        sort_cut_dad = np.argsort(cut_gen_dad)

        # Sử dụng mảng chỉ số ứng sang mảng sắp xếp của mẹ
        for idx, (d, m) in enumerate(zip(sort_cut_dad, sort_cut_mom)):
            gene_child_1[idx+pos1] = cut_gen_mom[d]
            gene_child_2[idx+pos1] = cut_gen_dad[m]

        return gene_child_1, gene_child_2

    def PMXCrossover(self, dad, mom):  # 3314.576

        # Chọn 2 điểm cắt ngẫu nhiên được sắp xếp tăng dần
        pos1, pos2 = sorted(random.sample(range(len(mom)), 2))
        # Tách ra đoạn gen của bố và mẹ
        gene_child_1 = np.zeros(len(mom), dtype=int)
        gene_child_2 = np.zeros(len(dad), dtype=int)
        # Lập ma trận ánh xạ bố-mẹ
        mapping_gene = np.vstack((dad[pos1:pos2], mom[pos1:pos2]))

        # Thiết lập ánh xạ
        for idx_p, (d, m) in enumerate(zip(dad, mom)):
            if idx_p in range(pos1, pos2):
                gene_child_1[idx_p] = mom[idx_p]
                gene_child_2[idx_p] = dad[idx_p]
                continue

            idx = idx_p
            # Ánh xạ phần gen của bố
            i = 1
            while d in mapping_gene[i]:
                idx = np.argwhere(mapping_gene[i] == d)[0][0]
                d = mapping_gene[np.abs(i-1)][idx]
            # Ánh xạ phần gen của mẹ
            i = 0
            while m in mapping_gene[i]:
                idx = np.argwhere(mapping_gene[i] == m)[0][0]
                m = mapping_gene[np.abs(i-1)][idx]

            gene_child_1[idx_p] = d
            gene_child_2[idx_p] = m

        return gene_child_1, gene_child_2

    def ArithmeticCrossover(self, dad, mom):  # 4241.379
        alpha = random.random()
        # Convert bố và mẹ về khoảng cách từ kho đến các điểm
        convert_dad = np.zeros(len(dad), dtype=float)
        convert_mom = np.zeros(len(mom), dtype=float)
        for idx, (d, m) in enumerate(zip(dad, mom)):
            convert_dad[idx] = np.linalg.norm(
                self.customers[0].xy_coord-self.customers[d].xy_coord)
            convert_mom[idx] = np.linalg.norm(
                self.customers[0].xy_coord-self.customers[m].xy_coord)

        # print(dad)
        # Tính child mới theo tọa độ
        child_1 = convert_dad + alpha * (convert_mom-convert_dad)
        child_2 = convert_mom + alpha * (convert_dad-convert_mom)

        gene_child_1 = child_1.copy()
        gene_child_2 = child_2.copy()

        sort_child_1 = np.sort(gene_child_1)  # sắp xếp tăng dần
        sort_child_2 = np.sort(gene_child_2)  # sắp xếp tăng dần

        for i in range(len(sort_child_1)):
            # gene_child[gene_child.index(sort_child[i])] = i
            gene_child_1[np.argwhere(
                gene_child_1 == sort_child_1[i])[0][0]] = i
            gene_child_2[np.argwhere(
                gene_child_2 == sort_child_2[i])[0][0]] = i

        # print(gene_child)
        gene_child_1 = gene_child_1.astype(int)
        gene_child_2 = gene_child_2.astype(int)

        for idx, (val1, val2) in enumerate(zip(gene_child_1, gene_child_2)):
            gene_child_1[idx] = self.sort_cluster[val1]
            gene_child_2[idx] = self.sort_cluster[val2]

        # print(gene_child)
        # exit()

        return gene_child_1, gene_child_2

    def PXCrossover(self, dad, mom):  # Bản chuẩn

        # Tách ra đoạn gen của bố và mẹ
        gene_child_1 = np.array(
            [mom[i] if random.randint(0, 1) == 1 else 0 for i in range(len(dad))])

        gene_child_2 = np.array(
            [dad[i] if random.randint(0, 1) == 1 else 0 for i in range(len(mom))])

        # Lọc ra các gen chưa được dùng đến
        filter_dad = [x for x in dad if x not in gene_child_1]
        filter_mom = [x for x in mom if x not in gene_child_2]

        # Lấp vào những chỗ trống còn lại
        gene_child_1 = np.array(
            [filter_dad.pop(0) if x == 0 else x for x in gene_child_1])
        gene_child_2 = np.array(
            [filter_mom.pop(0) if x == 0 else x for x in gene_child_2])

        return gene_child_1, gene_child_2

    def IPXCrossover(self, dad, mom):  # Bản chuẩn

        # Tách ra đoạn gen của bố và mẹ
        gene_child_1 = np.array(
            [mom[i] if random.randint(0, 1) == 1 else 0 for i in range(len(dad))])

        gene_child_2 = np.array(
            [dad[i] if random.randint(0, 1) == 1 else 0 for i in range(len(mom))])

        # Lọc ra các gen chưa được dùng đến
        filter_dad = list(reversed([x for x in dad if x not in gene_child_1]))
        filter_mom = list(reversed([x for x in mom if x not in gene_child_2]))

        # Lấp vào những chỗ trống còn lại
        gene_child_1 = np.array(
            [filter_dad.pop(0) if x == 0 else x for x in gene_child_1])
        gene_child_2 = np.array(
            [filter_mom.pop(0) if x == 0 else x for x in gene_child_2])

        return gene_child_1, gene_child_2

    def BestCostRouteCrossover(self, dad, mom):
        # Tách các route con từ các lộ trình cha và mẹ
        route_dad = self.individualToRoute(dad)
        route_mom = self.individualToRoute(mom)

        # Chọn ngẫu nhiên sub_route từ cha và mẹ 
        sub_route_mom = route_mom[np.random.randint(0, len(route_mom))]
        sub_route_dad = route_dad[np.random.randint(0, len(route_dad))]

        # Tráo đổi cho nhau và tiến hành xóa những phần tử đấy
        gene_child_1 = np.setdiff1d(dad, sub_route_mom)
        gene_child_2 = np.setdiff1d(mom, sub_route_dad)

        # Sau đó lấp lại bằng cách dùng phương pháp tham lam
        def greedySearch(intersect_gene, diff_gene):
            commom_part = np.copy(intersect_gene)
            # Thực hiện thuật toán tham lam
            # diff_gene: đoạn gen lắp vào
            # common_part: đoạn gen trả về
            for gen in diff_gene:

                parts = [self.cal_fitness_individualV2(np.insert(commom_part, idx, gen))[
                    0] for idx in range(len(commom_part)+1)]
                idx_cus_min = np.argmin(parts)

                # Chèn khách hàng
                commom_part = np.insert(commom_part, idx_cus_min, gen)
            return commom_part

        gene_child_1 = greedySearch(gene_child_1, sub_route_mom)
        gene_child_2 = greedySearch(gene_child_2, sub_route_dad)

        return gene_child_1, gene_child_2

    def STPBCrossover(self, dad, mom):  # 3411.464
        # probabilities = [0.1,0.3,0.3,0.1,0.2]
        probabilities = [0.25, 0.25, 0.25, 0.25]

        choice = np.random.choice(
            [i for i in range(len(probabilities))], p=probabilities)
        match choice:
            case 0:
                gene_child_1, gene_child_2 = self.heuristic_SinglePointCrossover(
                    dad, mom)
            case 1:
                gene_child_1, gene_child_2 = self.heuristic_TwoPointCrossoverV1(
                    dad, mom)  # Order
            case 2:
                gene_child_1, gene_child_2 = self.PMXCrossover(dad, mom)
            case 3:
                gene_child_1, gene_child_2 = self.BestCostRouteCrossover(dad, mom)
                    

        return gene_child_1, gene_child_2

    def sort_cluster_by_timewindow(self, cluster):
        # Convert giải pháp và danh sách cụm về khoảng cách từ kho đến các điểm
        distance_cluster = np.zeros(len(cluster), dtype=float)
        start_time_cluster = np.zeros(len(cluster), dtype=float)

        for idx, c in enumerate(cluster):
            distance_cluster[idx] = np.linalg.norm(
                self.customers[0].xy_coord-self.customers[c].xy_coord)
            start_time_cluster[idx] = self.customers[c].readyTime
        # print(start_time_cluster)
        # Convert khoảng cách về danh sách cluster được sắp xếp theo thời gian bắt đầu phuc vụ tăng dần nếu giống nhau thì theo khoảng cách
        sort_cluster = np.lexsort((start_time_cluster, distance_cluster))
        # print(sort_cluster)
        for i in range(len(sort_cluster)):
            sort_cluster[i] = cluster[sort_cluster[i]]

        return sort_cluster

    def re_cluster_by_timewindow(self, clusters):

        def check_concatenate(cluster1, cluster2):
            total_cluster = np.concatenate((cluster1, cluster2), axis=0)
            # Tính thời gian phục vụ cho toàn bộ khách hàng
            check  = sum([self.customers[i].serviceTime for i in total_cluster])

            # Tính trọng tải của toàn bộ khách hàng
            check_capacity = sum([self.customers[i].demand for i in total_cluster])

            # Kiểm tra điều kiện ràng buộc trọng tải
            if check_capacity > self._vehcicle_capacity:
                return False
            cluster1 = [self.customers[i].xy_coord for i in cluster1]
            cluster2 = [self.customers[i].xy_coord for i in cluster2]

            # Kết hợp hai cụm
            total_cluster = np.concatenate((cluster1, cluster2), axis=0)

            # Tính khoảng cách giữa các điểm trong cluster1 và cluster2
            distance = distance_cdist(total_cluster, total_cluster, metric='euclidean')

            # Tính khoảng cách trung bình giữa 2 điểm khách hàng trong cụm ghép
            aver_dist = np.mean(np.nonzero(distance))

            # Tính khoảng cách từ depot đến các điểm trong cụm ghép
            distance_to_depot = distance_cdist(
                total_cluster, [self.customers[0].xy_coord], metric='euclidean')
            
            # Khoảng cách trung bình từ kho đến cụm ghép
            avg_distance_to_depot = np.min(distance_to_depot)
            
            # Thời gian di chuyển ước lượng = 2* khoảng cách trung bình từ kho đến cụm + (số lượng khách hàng -1) * khoảng cáchtrung bình giữa 2 điểm + thời gian phục vụ toàn bộ khách hàng
            # số lượng khách hàng - 1 là số lượng cạch

            check += 2 * avg_distance_to_depot + (len(total_cluster)-1) * aver_dist
            return check <= self.customers[0].dueTime
        
        def concatenate_arrays(array, index1, index2):
            # Nối hai mảng theo chỉ số đã cho
            return [np.concatenate((array[index1], array[index2])).tolist()] + [array[i] for i in range(len(array)) if i != index1 and i != index2]
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

    def sort_cluster_by_distance(self, cluster):
        # Convert giải pháp và danh sách cụm về khoảng cách từ kho đến các điểm
        convert_cluster = np.zeros(len(cluster), dtype=float)

        for idx, c in enumerate(cluster):
            convert_cluster[idx] = np.linalg.norm(
                self.customers[0].xy_coord-self.customers[c].xy_coord)

        # Convert khoảng cách về danh sách cluster được sắp xếp theo khoảng cách
        sort_cluster = np.argsort(convert_cluster)

        for i in range(len(sort_cluster)):
            sort_cluster[i] = cluster[sort_cluster[i]]

        return sort_cluster

    def mutation(self, child):
        child_new = np.copy(child)

        pos1, pos2, pos3, pos4 = sorted(random.sample(range(len(child)), 4))
        # child_new = child[:pos1] + child[pos3:pos4+1] + child[pos2+1:pos3] + child[pos1:pos2+1] + child[pos4+1:]
        child_new = np.concatenate(
            (child[:pos1], child[pos3:pos4+1], child[pos2+1:pos3], child[pos1:pos2+1], child[pos4+1:]))

        return child_new

    def hybird(self):
        # Lấy vị trí bắt đầu của cá thể được phép lai ghép trong quần thể
        index = math.floor(self._conserve_rate*self._individual)

        while (len(self.__population) < self._individual):
            # Lấy ra tỉ lệ sinh của quần thể lần này
            hybird_rate = random.random()

            # Lấy ngẫu nhiên ra 2 cá thể đem trao đổi chéo
            dad, mom = random.sample(self.__population[index:], 2)

            # Kiểm tra tỉ lệ sinh so với tủi lệ trao đổi chéo
            if (hybird_rate > self._crossover_rate):
                continue

            # Tiến hành trao đổi chéo
            gene_child_1, gene_child_2 = self.STPBCrossover(
                np.copy(dad.customerList), np.copy(mom.customerList))

            # Kiếm tra tỉ lệ sinh với tỉ lệ đột biến gen
            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)

            child1 = Individual(customerList=gene_child_1)
            # child1.fitness,child1.distance = self.cal_fitness_individualV2(gene_child_1)
            self.__population.append(child1)

            if (len(self.__population) < self._individual):
                child2 = Individual(customerList=gene_child_2)
                # child2.fitness,child2.distance = self.cal_fitness_individualV2(gene_child_2)
                self.__population.append(child2)

    def fit(self, cluster):
        _start_time = time.time()
        self.initialPopulation(cluster)

        for _ in range(self._generation):
            self.cal_fitness_population()
            self.selection()
            self.hybird()
        self.process_time += round_float(time.time() - _start_time)

        self.cal_fitness_population()
        self.selection()

        self.best_fitness_global += self.__population[0].fitness
        self.best_distance_global += self.__population[0].distance
        best_route = self.individualToRoute(self.__population[0].customerList)
        self.best_route_global.append(best_route)
        self.route_count_global += len(best_route)
        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

    def fit_allClusters(self, clusters):
        # clusters = [[12, 13, 14, 15, 16, 17, 18, 19, 77, 82, 83, 84, 86, 87, 88, 90, 91, 96], [41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 40, 53, 54, 55, 56, 57, 58, 59, 60], [20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 70, 71, 73, 76, 78, 79, 80, 81, 85], [1, 3, 4, 7, 89, 92, 94, 95, 97, 98, 99, 100, 2, 5, 8, 9, 10, 11, 75, 93], [6, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 74]]

        clusters = self.re_cluster_by_timewindow(clusters)
        # print('Số cụm sau đi phân cụm lại',len(clusters),clusters)
        # exit()
        for i in range(len(clusters)):
            self.fit(cluster=clusters[i])
        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global , self.process_time


if __name__ == "__main__":
    # Thông số K-means
    N_CLUSTER = 20
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100
    # Thông số GA
    INDIVIDUAL = 100
    GENERATION = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    VEHCICLE_CAPACITY = 700
    CONSERVE_RATE = 0.1
    M = 0
    # Thông số ngoài
    DATA_ID = "R101"  # File dữ liệu cụ thể
    DATA_NAME = "R1"  # Bộ dữ liệu
    DATA_NUMBER_CUS = "100"  # Số lượng khách hàng
    RUN_TIMES = 10  # Số lượng chạy
    EXCEL_FILE = None  # File excel xuất ra kết quả bộ dữ liệu
    FILE_EXCEL_PATH = "result/"
    FILE_NAME = "_TestKmeans"
    # ================================================================================================================
    if (DATA_ID != None):
        url_data = "data/txt/"+DATA_NUMBER_CUS+"/"+DATA_ID[:-2]+"/"
        data_files = [DATA_ID+".txt"]
    else:
        url_data = "data/txt/"+DATA_NUMBER_CUS+"/"+DATA_NAME+"/"
        data_files = sorted(
            [f for f in os.listdir(url_data) if f.endswith(('.txt'))])
        EXCEL_FILE = FILE_EXCEL_PATH + DATA_NAME + FILE_NAME + ".xlsx"

    len_data = len(data_files)
    print(f"Bộ dữ liệu {DATA_NAME}: {data_files}")
    run_time_data = 0
    route_count_data = 0
    distance_data = 0
    fitness_data = 0

    C_scope = range(1, 10)
    data_excel = []
    for data_file in data_files:
        run_time_mean = 0
        route_count_mean = 0
        distance_mean = 0
        fitness_mean = 0

        _start_time = time.time()
        # Khởi tạo dữ liệu
        data, customers = load_txt_dataset(
            url=url_data, name_of_id=data_file)
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))

        print("#K-means =============================")
        # chạy kmeans
        _start_time = time.time()
        kmeans = Kmeans(epsilon=EPSILON, maxiter=MAX_ITER, n_cluster=N_CLUSTER)
        warehouse = data[0]
        data_kmeans = np.delete(data, 0, 0)
        # U1, V1, step = kmeans.k_means(data_kmeans)
        # print("Thời gian chạy K-means:", round_float(time.time() - _start_time))

        # cluster = kmeans.data_to_cluster(U1)

        U1, V1, step = kmeans.k_means_lib_sorted(data_kmeans,warehouse)
        cluster = kmeans.data_to_cluster(U1)
        print(cluster)
        # kmeans.elbow_k_means_lib(data_kmeans,C_scope)
        # kmeans.elbow_k_means(data_kmeans,C_scope)
        # -------------------------------------------------------------------------------------------------------------------------------------

        print("#GA =============================")

        for j in range(RUN_TIMES):

            ga = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    vehcicle_capacity=VEHCICLE_CAPACITY, conserve_rate=CONSERVE_RATE, M=M, customers=customers)
            
            best_fitness_global, best_route_global, best_distance_global, route_count_global, process_time = ga.fit_allClusters(
                clusters=cluster)

            run_time_mean += process_time

            print(f"Thời gian chạy {data_file[:-4]} lần", j+1, ":", process_time)
            print("Fitness: ", round_float(best_fitness_global))
            print("Distance: ", round_float(best_distance_global))
            print("Số lượng route: ", route_count_global)
            print(best_route_global)
            route_count_mean += route_count_global
            distance_mean += best_distance_global
            fitness_mean += best_fitness_global
            data_excel += [[route_count_global,
                            round_float(best_distance_global), process_time]]
            print("===================================")
        # Thống kê file dữ liệu

        print(f"#Thống kê {data_file[:-4]} =============================")
        print("Số lượt chạy mỗi bộ dữ liệu ", RUN_TIMES)
        print("Fitness trung bình: ", round_float(fitness_mean/RUN_TIMES))
        print("Số lượng route trung bình: ",
              round_float(route_count_mean/RUN_TIMES))
        print("Thời gian di chuyển trung bình: ",
              round_float(distance_mean/RUN_TIMES))
        print("Thời gian chạy trung bình: ",
              round_float(run_time_mean/RUN_TIMES))
        run_time_data += round_float(run_time_mean/RUN_TIMES)
        route_count_data += round_float(route_count_mean/RUN_TIMES)
        distance_data += round_float(distance_mean/RUN_TIMES)
        fitness_data += round_float(fitness_mean/RUN_TIMES)
        data_excel += [[round_float(route_count_mean/RUN_TIMES),
                        round_float(distance_mean/RUN_TIMES)]]
        print("====================================================================================================================")

    # Thống kê data
    print("=====================================================================================================================================")
    print(f"#Thống kê {DATA_NAME} =============================")
    print("Số lượt chạy mỗi bộ dữ liệu ", RUN_TIMES)
    print("Fitness trung bình: ", round_float(fitness_data/len_data))
    print("Số lượng route trung bình: ", round_float(route_count_data/len_data))
    print("Thời gian di chuyển trung bình: ",
          round_float(distance_data/len_data))
    print("Thời gian chạy trung bình: ", round_float(run_time_data/len_data))

    title_name = ['Route', 'Distance', 'RunTime']
    # if (DATA_NAME != None):
    #     # write_excel_file(data_excels=data_excel,data_files=data_files,run_time=RUN_TIMES,alogirthm='GA',fileio=EXCEL_FILE)
    #     write_excel_file(data_excels=data_excel,data_files=data_files,run_time=RUN_TIMES,alogirthms='GA',title_names=title_name,fileio=EXCEL_FILE)

    # count = 1
    # for i in range(len(best_route)):
    #     for j in range(len(best_route[i])):
    #         print('route ',count,best_route[i][j])
    #         count+=1
    # print('best_fitness',best_fitness)
