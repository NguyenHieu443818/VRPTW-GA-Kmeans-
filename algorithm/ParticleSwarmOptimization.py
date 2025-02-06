import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
from ultility.utilities import round_float,distance_cdist
import os
import time

class Particle:                     # định nghĩa 1 hạt particle
    def __init__(self, num_customers):  #hàm khởi tạo lớp particle có 1 tham số số KH
        self.position = np.random.uniform(-5, 5, size=num_customers)    #Khởi tạo Vị trí ngẫu nhiên của hạt có giá trị từ -5 đến 5
        self.velocity = np.random.uniform(-1, 1, size=num_customers)    #Khởi tạo Vận tốc ngẫu nhiên của hạt có giá trị từ -1 đến 1
        self.pbest_position = self.position.copy()  #Biến để lưu vị trí tốt nhất mà hạt từng đạt được, ban đầu vị trí tốt nhất là vị trí được khởi tạo
        self.pbest_fitness = float('inf')           #Biến lưu giá trị fitness tốt nhất  mà hạt từng đạt được, lúc đầu khởi tạo là vô cùng lớn
        self.pbest_distance = 0

    def print(self):
        print(self.position,' ',self.velocity)

class PSO():
    def __init__(self,num_particles,max_iter,vehcicle_capacity,M,w,c1,c2,customers: list=None):
        self.num_particles = num_particles #số hạt trong đàn
        self.max_iter = max_iter # Số vòng lặp
        self._vehcicle_capacity = vehcicle_capacity # Trọng tải của xe
        self._M = M # Sai số M
        self.customers=customers # Danh sách khách hàng
        self.w = w # Hệ số quán tính của PSO
        self.c1 = c1 # Hệ số học cá nhân
        self.c2 = c2 # Hệ số học xã hội
        self.gbest_postition = 0 # Vị trí tốt nhất của quần thể
        self.gbest_fitness = 0 # Độ thích nghi tốt nhất của quần thể
        self.gbest_distance = 0 # Khoảng cách di chuyển tốt nhất của quần thể

        self.best_distance_global = 0  # Khoảng cách tốt nhất sau khi tổng hợp các cụm
        self.route_count_global = 0 # Số lộ trình tốt nhất sau khi tổng hợp các cụm
        self.best_fitness_global = 0 # Độ thích nghi tốt nhất sau khi tổng hợp các cụm
        self.best_route_global = [] # Lộ trình tốt nhất sau khi tổng hợp các cụm

    
    def initialPopulation(self,cluster)->np.ndarray:
        self.__population = [Particle(len(cluster)) for _ in range(self.num_particles)]

     # Tách thành các lộ trình con
    def individualToRoute(self,individual):
        route = [] #Lộ trình tổng
        vehicle_load = 0 #trọng tải xe
        sub_route = [] #Lộ trình con
        elapsed_time = 0 #Mốc thời gian hiện tại của xe
        last_customer_id = 0 #Vị khách đã xét trước đó 
        for customer_id in individual:
            customer = self.customers[customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand 

            #Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            moving_time = np.linalg.norm(customer.xy_coord-self.customers[last_customer_id].xy_coord) 

            #Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M - elapsed_time - moving_time, 0) 

            #Thời gian di chuyển từ điểm đang xét về kho
            return_time = np.linalg.norm(customer.xy_coord-self.customers[0].xy_coord)

            update_elapsed_time = elapsed_time + service_time + moving_time + waiting_time + return_time

            if(update_vehicle_load <= self._vehcicle_capacity)and(update_elapsed_time <= self.customers[0].dueTime + self._M):    
                sub_route.append(customer_id)
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
                elapsed_time =  service_time + customer.readyTime
            last_customer_id = customer_id
        if sub_route != []:
            # Lưu lộ trình con còn lại sau khi xét hết các điểm
            route.append(sub_route)

        return route # 1 list các array

  

    #Tính mức độ thích nghi trên toàn bộ quần thể
    def cal_fitness_population(self,cluster)->np.ndarray:
        for individual in self.__population:
            fitness,distance = self.cal_fitness_individualV2(individual.position,cluster)
            # if (individual.pbest_distance >= distance):
            if (individual.pbest_fitness >= fitness):

                individual.pbest_fitness = fitness
                individual.pbest_distance = distance

        # print(self.__population)

    def cal_fitness_individual(self,individual,cluster):
        individual = self.convert_customer(individual,cluster)
        route = self.individualToRoute(individual)
        fitness = 0
        distance = 0
        for sub_route in route:
            sub_route_time_cost = 0 # Thời gian đợi và phạt của 1 route con
            sub_route_distance = 0 # Thời gian di chuyển của 1 route con
            elapsed_time = 0
            last_customer_id = 0
            for customer_id in sub_route:
                customer = self.customers[customer_id]

                # Thời gian di chuyển giữa 2 điểm
                moving_time = np.linalg.norm(customer.xy_coord-self.customers[last_customer_id].xy_coord)

                #Cập nhật thời gian di chuyển
                sub_route_distance += moving_time

                # Mốc thời gian đến khách hàng thứ customer_id
                arrive_time =  moving_time + elapsed_time 

                #Thời gian chờ đợi nếu xe đang ở thời gian chưa đến thời gian bắt đầu phục vụ
                waiting_time = max(customer.readyTime - self._M - arrive_time, 0) 

                #Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
                delay_time  = max (arrive_time - customer.dueTime - self._M, 0) 

                #Cập nhật thời gian đợi và phạt
                sub_route_time_cost += waiting_time + delay_time

                #Cập nhật mốc thời gian mới của xe
                elapsed_time =  arrive_time + customer.serviceTime + waiting_time

                #Cập nhật khách hàng đang xét
                last_customer_id = customer_id

            #Thời gian di chuyển từ điểm cuối cùng về kho
            return_time = np.linalg.norm(self.customers[last_customer_id].xy_coord-self.customers[0].xy_coord)
            sub_route_distance += return_time
            # print('sub_route_distance',sub_route_distance)
            fitness += sub_route_distance + sub_route_time_cost
            distance += sub_route_distance
        return fitness,distance
   
    def cal_fitness_individualV2(self,individual,cluster):
        individual = self.convert_customer(individual,cluster)

        vehicle_load = 0 #trọng tải xe
        elapsed_time = 0 #Mốc thời gian hiện tại của xe
        last_customer_id = 0 #Vị khách đã xét trước đó 
        sub_route = [] #Lộ trình con
        fitness = 0
        distance = 0
        depot = self.customers[0] # Lấy dữ liệu kho
        for customer_id in individual:
            customer = self.customers[customer_id]
            last_customer = self.customers[last_customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand 

            #Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            moving_time = np.linalg.norm(customer.xy_coord-last_customer.xy_coord)

            # Mốc thời gian đến khách hàng thứ customer_id
            arrive_time =  moving_time + elapsed_time 

            #Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M - arrive_time, 0)

            #Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
            delay_time  = max (arrive_time - customer.dueTime - self._M, 0)

            #Thời gian di chuyển từ điểm đang xét về kho
            return_time = np.linalg.norm(customer.xy_coord-depot.xy_coord)

            update_elapsed_time = arrive_time + service_time + waiting_time + return_time

            if(update_vehicle_load <= self._vehcicle_capacity)and(update_elapsed_time <= depot.dueTime + self._M):
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
                sub_route.append(customer_id)
                #Cập nhật thời gian di chuyển
                distance += moving_time
                #Cập nhật thời gian đợi và phạt
                fitness += waiting_time + delay_time
            else:
                distance += return_time + np.linalg.norm(last_customer.xy_coord-depot.xy_coord)
                sub_route = [customer_id]
                # Cập nhật khoảng cách di chuyển từ điểm kết thúc về kho và bắt đầu từ kho đến điểm (Không cần phải tính phạt vì sẽ là bị dữ liệu sai)
                waiting_time = max(customer.readyTime - self._M - return_time, 0) 
                fitness +=  waiting_time
                vehicle_load = demand
                elapsed_time =  service_time + return_time +  waiting_time

            # print(fitness)
            last_customer_id = customer_id
            
        if sub_route != []:
            distance += np.linalg.norm(self.customers[last_customer_id].xy_coord-depot.xy_coord)
            fitness += distance


        return fitness,distance

    def update_gbest(self):
        gbest = min(self.__population, key=lambda x:x.pbest_fitness)
        # gbest = min(self.__population, key=lambda x:x.pbest_distance)

        self.gbest_fitness = gbest.pbest_fitness
        self.gbest_postition = gbest.pbest_position.copy()
        self.gbest_distance = gbest.pbest_distance

    # Cập nhật vận tốc mới
    def update_velocity(self, particle):
        inertia_influence = self.w * particle.velocity
        personal_influence = self.c1 * np.random.rand() * (particle.pbest_position - particle.position)
        social_influence  = self.c2 * np.random.rand() * (self.gbest_postition - particle.position)
        return inertia_influence + personal_influence + social_influence

    def update_population(self):
        for individual in self.__population:
            individual.velocity = self.update_velocity(individual)
            self.update_position(individual)

    #Cập nhật vị trí mới
    def update_position(self,particle):
        particle.position += particle.velocity
    
    #biến đổi DS giá trị giải pháp thành danh sách chỉ số 
    def convert_customer(self,solution,cluster):
        temp_solution = solution.copy() #copy dữ liệu gốc ra biến temp_solution

        sorted_solution = np.sort(temp_solution) #sắp xếp giải pháp tăng dần

        for i in range(len(temp_solution)):
            # temp_solution[i] = sorted_solution.index(temp_solution[i])+1
            # temp_solution[i] = np.argwhere(sorted_solution == temp_solution[i])[0][0] +1
            temp_solution[np.argwhere(temp_solution == sorted_solution[i])[0][0]] = i

        # temp_solution = [sorted_solution.index(i) + 1 for i in temp_solution]
        temp_solution = temp_solution.astype(int)   #ép kiểu số nguyên
        for idx,val in enumerate(temp_solution):
            temp_solution[idx] = cluster[val] 
  
        return temp_solution
    
    def sort_cluster_by_distance(self,cluster):
        # Convert giải pháp và danh sách cụm về khoảng cách từ kho đến các điểm
        convert_cluster = np.zeros(len(cluster),dtype=float)

        for idx, c in enumerate( cluster):    
            convert_cluster[idx] = np.linalg.norm(self.customers[0].xy_coord-self.customers[c].xy_coord)

        # Convert khoảng cách về danh sách cluster được sắp xếp theo khoảng cách
        sort_cluster = np.argsort(convert_cluster)
        
        for i in range(len(sort_cluster)):
            sort_cluster[i] = cluster[sort_cluster[i]]
        
        return sort_cluster

    def re_cluster_by_timewindow(self, clusters):

        def check_concatenate(cluster1, cluster2):
            total_cluster = np.concatenate((cluster1, cluster2), axis=0)
            check  = sum([self.customers[i].serviceTime for i in total_cluster])
            check_capacity = sum([self.customers[i].demand for i in total_cluster])
            if check_capacity > self._vehcicle_capacity:
                return False
            cluster1 = [self.customers[i].xy_coord for i in cluster1]
            cluster2 = [self.customers[i].xy_coord for i in cluster2]
            # # Tính khoảng cách giữa các điểm trong cluster1 và cluster2

            # Kết hợp hai cụm
            total_cluster = np.concatenate((cluster1, cluster2), axis=0)
            # Tính khoảng cách giữa các điểm trong cluster1 và cluster2
            distance = distance_cdist(total_cluster, total_cluster, metric='euclidean')
            aver_dist = np.mean(np.nonzero(distance))
            # Tính khoảng cách từ depot đến các điểm trong cluster1 và cluster2
            distance_to_depot = distance_cdist(
                total_cluster, [self.customers[0].xy_coord], metric='euclidean')
            avg_distance_to_depot = np.min(distance_to_depot)

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
        # print(concatenate_arrays(clusters, 0, 4)[0])
        # print(clusters[8])
        # return check_concatenate(concatenate_arrays(clusters, 0, 4)[0],clusters[8])
        return clusters

    def fit(self, cluster):
        # Khởi tạo quần thể

        self.initialPopulation(cluster)
        # print([individual.print() for individual in self.__population])
        # print([len(individual.position) for individual in self.__population])
        # Sắp xếp lại danh sách cluster theo khoảng cách 
        # cluster = self.sort_cluster_by_distance(cluster)
        for i in range(self.max_iter):
            # Tính fitness và cập nhật pbest
            _start_time = time.time()
            self.cal_fitness_population(cluster)
            
            # Tính global best
            self.update_gbest()
            # Cập nhật vận tốc và vị trí cho quần thể
            self.update_population()
            # print('loop',i)
            # print("Thời gian chạy :", round_float(time.time() - _start_time))
            # exit()
        print(self.convert_customer(self.gbest_postition,cluster))
            

        self.best_fitness_global += self.gbest_fitness
        self.best_distance_global += self.gbest_distance
        best_route= self.individualToRoute(self.convert_customer(self.gbest_postition,cluster))
        self.best_route_global.append(best_route)
        self.route_count_global += len(best_route)

    def fit_allClusters(self, clusters):
        clusters = self.re_cluster_by_timewindow(clusters)
        print('Số cụm sau đi phân cụm lại',len(clusters),clusters)
        for i in range(len(clusters)):
            self.fit(cluster=clusters[i])
        return self.best_fitness_global,self.best_route_global,self.best_distance_global,self.route_count_global

if __name__ == "__main__":
    import time
    #Thông số K-means
    N_CLUSTER = 6
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100
    #Thông số PSO
    INDIVIDUAL = 200
    MAX_ITER_PSO = 100
    W = 0.8
    C1 = 0.15
    C2 = 0.1
    VEHCICLE_CAPACITY = 200
    M = 0
    #Thông số dữ liệu
    DATA_ID = None #File dữ liệu cụ thể
    DATA_NAME = "R1" #Bộ dữ liệu
    RUN_TIMES = 10
    #================================================================================================================
    if (DATA_ID != None):
        url_data = "data/csv/"+DATA_ID[:-2]+"/"
        data_files = [DATA_ID+".csv"]
        
    else:
        url_data = "data/csv/"+DATA_NAME+"/"
        data_files = [f for f in os.listdir(url_data) if f.endswith(('.csv'))]

    len_data = len(data_files)
    print(f"Bộ dữ liệu {DATA_NAME}: {data_files}")
    run_time_data = 0
    route_count_data = 0
    distance_data = 0
    fitness_data = 0
    C_scope = range(1,10)
    for data_file in data_files:
        run_time_mean = 0
        route_count_mean = 0
        distance_mean = 0
        fitness_mean = 0
        _start_time = time.time()
        #Khởi tạo dữ liệu
        data,customers = load_txt_dataset(url=url_data,name_of_id=data_file,number_of_customer=NUMBER_OF_CUSTOMER)
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))

        print("#K-means =============================")
        #chạy kmeans
        _start_time = time.time() 
        kmeans = Kmeans(epsilon=EPSILON, maxiter=MAX_ITER,n_cluster=N_CLUSTER)
        data_kmeans = np.delete(data,0,0)
        U1, V1, step = kmeans.k_means(data_kmeans)
        print("Thời gian chạy K-means:", round_float(time.time() - _start_time))

        cluster = kmeans.data_to_cluster(U1)
        # kmeans.elbow_k_means(data_kmeans,C_scope)

        #-------------------------------------------------------------------------------------------------------------------------------------

        print("#PSO =============================")


        for j in range(RUN_TIMES):
            
            pso = PSO(num_particles= INDIVIDUAL,max_iter=MAX_ITER_PSO,vehcicle_capacity=VEHCICLE_CAPACITY,M=M,w=W,c1=C1,c2=C2,customers=customers)
            _start_time = time.time()
            best_fitness_global,best_route_global,best_distance_global,route_count_global = pso.fit_allClusters(clusters=cluster)
            run_time = round_float(time.time() - _start_time)
            
            run_time_mean += run_time
            
            print(f"Thời gian chạy {data_file[0:4]} lần",j+1,":", run_time)
            print("Fitness: ", round_float(best_fitness_global))
            print("Distance: ", round_float(best_distance_global))
            print("Số lượng route: ",route_count_global)
            print(best_route_global)
            route_count_mean += route_count_global
            distance_mean += best_distance_global
            fitness_mean += best_fitness_global
            print("===================================")
        #Thống kê file dữ liệu

        print(f"#Thống kê {data_file[0:4]} =============================")
        print("Số lượt chạy mỗi bộ dữ liệu ",RUN_TIMES)
        print("Fitness trung bình: ", round_float(fitness_mean/RUN_TIMES))
        print("Số lượng route trung bình: ", round_float(route_count_mean/RUN_TIMES))
        print("Thời gian di chuyển trung bình: ", round_float(distance_mean/RUN_TIMES))
        print("Thời gian chạy trung bình: ",round_float(run_time_mean/RUN_TIMES))
        run_time_data += round_float(run_time_mean/RUN_TIMES)
        route_count_data += round_float(route_count_mean/RUN_TIMES)
        distance_data += round_float(distance_mean/RUN_TIMES)
        fitness_data += round_float(fitness_mean/RUN_TIMES)
        print("====================================================================================================================")

    #Thống kê data   
    print("=====================================================================================================================================")
    print(f"#Thống kê {DATA_NAME} =============================")
    print("Số lượt chạy mỗi bộ dữ liệu ",RUN_TIMES)
    print("Fitness trung bình: ", round_float(fitness_data/len_data))
    print("Số lượng route trung bình: ", round_float(route_count_data/len_data))
    print("Thời gian di chuyển trung bình: ", round_float(distance_data/len_data))
    print("Thời gian chạy trung bình: ",round_float(run_time_data/len_data))