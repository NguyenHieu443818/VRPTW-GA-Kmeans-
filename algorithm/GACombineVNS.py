import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.GeneticAlgorithm import GA, Individual
from algorithm.VariableNeighborhoodSearchAlgorithm import VNS
from algorithm.kmeans import Kmeans
import math
import random
from ultility.utilities import round_float, write_excel_file, distance_cdist, create_graph
import os
import time

C_ID = 0
C_X = 1
C_Y = 2
C_DEMAND = 3
C_READY_TIME = 4
C_DUE_TIME = 5
C_SERVICE_TIME = 6

class GA_VNS(GA, VNS):
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.15, vehicle_capacity: float = 200, conserve_rate: float = 0.1, M: float = 50, customers: list = None, graph_data: np.ndarray = None, list_n_l: list = None, beta_0: int = 0, beta_1: int = 0):
        GA.__init__(self, individual=individual, generation=generation, crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                    vehicle_capacity=vehicle_capacity, conserve_rate=conserve_rate, M=M, customers=customers, graph_data=graph_data)
        VNS.__init__(self, list_n_l=list_n_l, beta_0=beta_0,beta_1=beta_1, generation=generation)
                     

    # Hàm kiểm tra xem chuỗi gen mới có tốt hơn chuỗi gen cũ
    def is_better(self, new_gene: list, gene: list):
        fitness_gene, distance_gene = GA.cal_fitness_individualV2(individual=np.array(gene, dtype=np.int64), customers=self.customers,
                                                      graph_data=self.graph_data, vehicle_capacity=self._vehicle_capacity, M=self._M)

        fitness_new_gene, distance_new_gene = GA.cal_fitness_individualV2(individual=np.array(new_gene, dtype=np.int64), customers=self.customers,
                                                          graph_data=self.graph_data, vehicle_capacity=self._vehicle_capacity, M=self._M, fitness_to_branch_bound=fitness_gene)
        return fitness_new_gene < fitness_gene

    # Tạm thời chưa tối ưu code -> đề xuất sửa thành thuật toán nhánh cận tăng tốc độ tính toán
    def generate_first_individual(self, clusters):
        # # Chuyển np.ndarray sang list để sort không bị lỗi
        # cluster = list(cluster)
        copy_cluster = clusters.copy()

        # Sắp xếp tăng dần theo dueTime, nếu bằng nhau thì theo readyTime
        copy_cluster.sort(key=lambda x:(self.customers[x, C_DUE_TIME], self.customers[x, C_READY_TIME]))
  
        new_individual = [copy_cluster[0]]
        del copy_cluster[0]

        while copy_cluster:
            cus_min = 0
            idx_min = 0
            fitness_min = float('inf')  # sửa thành 'inf' vì đang tìm min

            for idx, cus in enumerate(copy_cluster):
                test = new_individual + [cus]  # tạo bản copy tạm để đánh giá
                fitness_part, _ = GA.cal_fitness_individualV2(individual=np.array(test, dtype=np.int64), customers=self.customers, graph_data=self.graph_data, vehicle_capacity=self._vehicle_capacity, M=self._M, fitness_to_branch_bound=fitness_min)
                    
                if fitness_part < fitness_min:  # cập nhật nếu tốt hơn
                    fitness_min = fitness_part
                    cus_min = cus
                    idx_min = idx

            # Chèn khách hàng tốt nhất
            new_individual.append(cus_min)
            del copy_cluster[idx_min]

        return new_individual

    # def initial_population(self, clusters) -> list:
        
    #     for _ in range(self._individual - 1):
    #         cluster_copy = [random.sample(cluster, len(cluster)) for cluster in clusters]
                            
    #         customer_list = [item for sublist in cluster_copy for item in sublist]
            
    #         self.population.append(Individual(customer_list=customer_list))

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
            gene_child_1, gene_child_2 = self.STPB_crossover(dad.customer_list, mom.customer_list)
                
            gene_child_3, gene_child_4 = self.STPB_crossover(dad.customer_list, self.population[0].customer_list)
                

            # Kiếm tra tỉ lệ sinh với tỉ lệ đột biến gen
            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)
                gene_child_3 = self.mutation(gene_child_3)
                gene_child_4 = self.mutation(gene_child_4)

            self.population.append(Individual(customer_list=gene_child_1))
            self.population.append(Individual(customer_list=gene_child_2))
            self.population.append(Individual(customer_list=gene_child_3))
            self.population.append(Individual(customer_list=gene_child_4))

            if (len(self.population) > self._individual):
                del self.population[self._individual:]
                break

    def fit(self, clusters):
        clusters = self.re_cluster_by_timewindow(clusters)
        print('clusters after re-cluster by timewindow: ', len(clusters),' ',clusters)
        routes =[]
        
        # self.population.append(Individual(customer_list= self.generate_first_individual(clusters)))
            
        _start_time = time.time()
        for cluster in clusters:
            self.initial_population(cluster)
            self.cal_fitness_population()
            for _ in range(self._generation):
                self.selection()
                self.hybird()

                self.cal_fitness_population()
                
                self.population.sort(key=lambda x: x.fitness)
                self.population[0].customer_list = VNS.fit(self, gene=self.population[0].customer_list)
                    
            self.process_time += round_float(time.time() - _start_time)
            self.cal_fitness_population()
            self.selection()
            routes.append(self.population[0].customer_list)
            self.best_fitness_global += self.population[0].fitness
            self.best_distance_global += self.population[0].distance
            best_route, _, _ = self.individual_to_route(self.population[0].customer_list)
            self.best_route_global.append(best_route)
            self.route_count_global += len(best_route)
            self.best_fitness_pM = -1
            self.best_fitness_pD = -1
            
        # self.process_time += round_float(time.time() - _start_time)    
        # route_total = [item for sublist in routes for item in sublist]
        # final_result = VNS.fit(self, gene=route_total)
        
        # self.best_fitness_global, self.best_distance_global = self.cal_fitness_individualV2(individual=np.array(final_result, dtype=np.int64), customers=self.customers, graph_data=self.graph_data, vehicle_capacity=self._vehicle_capacity, M=self._M)
        # best_route, _, _ = self.individual_to_route(final_result)
        # self.best_route_global.append(best_route)
        # self.route_count_global += len(best_route)   

        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global, self.process_time
