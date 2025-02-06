import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
from ultility.utilities import round_float
import os
from functools import reduce
import time

class Ant():
    def __init__(self, customerList:list=None,fitness:np.float64=0,distance:np.float64=0):
        self.customerList=customerList
        self.fitness=fitness
        self.distance=distance

    def print(self):
        print(self.customerList,' ',self.fitness,' ',self.distance)

class ACO():
    def __init__(self,num_ants,max_iter,vehcicle_capacity,M,alfa,beta,sigm,ro,th,customers:list=None):
        self.num_ants = num_ants #số lượng kiến
        self.max_iter = max_iter # Số vòng lặp
        self._vehcicle_capacity = vehcicle_capacity # Trọng tải của xe
        self._M = M # Sai số M
        self.customers=customers # Danh sách khách hàng
        self.alfa = alfa # Hệ số ảnh hưởng nồng độ pheromone
        self.beta = beta # Hệ số ảnh hưởng khoảng cách di chuyển
        self.sigm =sigm # Hệ số trong quá trình update feromone (Chưa rõ)
        self.ro = ro # Hệ số trong quá trình update feromone (Chưa rõ)
        self.th=th # Hệ số trong quá trình update feromone (Chưa rõ)

        self.gbest_postition = 0 # Vị trí tốt nhất của quần thể
        self.gbest_fitness = 0 # Độ thích nghi tốt nhất của quần thể
        self.gbest_distance = 0 # Khoảng cách di chuyển tốt nhất của quần thể

        self.best_distance_global = 0  # Khoảng cách tốt nhất sau khi tổng hợp các cụm
        self.route_count_global = 0 # Số lộ trình tốt nhất sau khi tổng hợp các cụm
        self.best_fitness_global = 0 # Độ thích nghi tốt nhất sau khi tổng hợp các cụm
        self.best_route_global = [] # Lộ trình tốt nhất sau khi tổng hợp các cụm


    def generateGraph(self,cluster):
        # Khởi tạo ma trận đồ thị của các khách hàng (ma trận tam giác)
        self.edges = { (min(a,b),max(a,b)) : np.linalg.norm(self.customers[a].xy_coord-self.customers[b].xy_coord) for a in cluster for b in cluster}
        # Khởi tạo ma trận pheromone của đồ thị (ma trận tam giác)
        self.feromones = { (min(a,b),max(a,b)) : 1 for a in cluster for b in cluster if a!=b }
        
    def initialPopulation(self,cluster)->np.ndarray:
        self.__population = [Ant(customerList=self.solutionOfOneAnt(cluster)) for _ in range(self.num_ants)]
    
    def cal_fitness_population(self)->np.ndarray:
        for individual in self.__population:
            fitness,distance = self.cal_fitness_individualV2(list(np.concatenate(individual.customerList)))
            individual.fitness = fitness
            individual.distance = distance

    def solutionOfOneAnt(self,cluster):
        solution = list()
        # Thiết lập kho 
        depot = self.customers[0]
        # Tạo vòng lặp để tìm các route con
        while(len(cluster)!=0):
            path = list()
            customer_id = np.random.choice(cluster)
            last_customer_id = 0
            customer = self.customers[customer_id]
            # Nhu cầu của khách hàng customer_id
            capacity = customer.demand

            #Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            moving_time = self.edges[min(last_customer_id, customer_id),max(last_customer_id, customer_id)]
            
            #Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M - moving_time, 0) 

            # Mốc thời gian sau khi di chuyển
            elapsed_time =  service_time + moving_time + waiting_time 
            
            last_customer_id = customer_id
            path.append(int(customer_id))
            cluster = cluster[cluster != customer_id]

            # Tạo vòng lặp xét các route con 1
            while(len(cluster)!=0):
                # Tính toán xác suất để kiến di chuyển từ điểm hiện tại (hàm map giúp ánh xạ từng phần tử của cluster đến customer_id)
                probabilities = list(map(lambda x: ((self.feromones[(min(x,customer_id), max(x,customer_id))])**self.alfa)*((1/self.edges[(min(x,customer_id), max(x,customer_id))])**self.beta), cluster))
                probabilities = probabilities/np.sum(probabilities)
                # Chọn khách hàng tiếp tiếp theo dựa theo danh sách xác suất
                customer_id = np.random.choice(cluster, p=probabilities)
                customer = self.customers[customer_id]

                # Cập nhật nhu cầu
                capacity += customer.demand

                # Thời gian di chuyển giữa 2 điểm
                moving_time = self.edges[min(last_customer_id, customer_id),max(last_customer_id, customer_id)]

                #Thời gian chờ đợi
                waiting_time = max(customer.readyTime - self._M - elapsed_time - moving_time, 0) 

                # Thời gian trở về kho
                return_time = self.edges[(0, customer_id)]
                # Mốc thời gian tiếp theo
                elapsed_time +=  moving_time+ waiting_time + customer.serviceTime + return_time  

                if(capacity <= self._vehcicle_capacity)and(elapsed_time <= depot.dueTime + self._M):    
                    last_customer_id = customer_id
                    path.append(int(customer_id))
                    cluster = cluster[cluster != customer_id]
                    elapsed_time -= return_time
                else:
                    break
            solution.append(path)
  
        return solution

    # Kiểu tính này đang có vấn đề sự ảnh hưởng của việc phạt của waiting và delay không nhiều
    def cal_fitness_individualV2(self,individual):
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
            moving_time = self.edges[min(last_customer_id, customer_id),max(last_customer_id, customer_id)]

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
            distance += self.edges[(0,last_customer_id)]
            fitness += distance


        return fitness,distance
    

    def updateFeromone(self): # Gây khó khăn vì ij tại mỗi vị trí trên tuyến đường nên có nồng độ pherômne khác nhau
        Lavg = reduce(lambda x,y: x+y, (i.fitness for i in self.__population))/len(self.__population)
        self.feromones = { k : (self.ro + self.th/Lavg)*v for (k,v) in self.feromones.items() }
        self.__population.sort(key = lambda x: x.fitness)
        if(self.bestSolution!=None):
            if(self.__population[0].fitness < self.bestSolution.fitness):
                self.bestSolution = self.__population[0]
            for path in self.bestSolution.customerList:
                for i in range(len(path)-1):
                    self.feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = self.sigm/self.bestSolution.fitness + self.feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
        else:
            self.bestSolution = self.__population[0]
        for l in range(self.sigm):
            paths = self.__population[l].customerList
            L = self.__population[l].fitness
            for path in paths:
                for i in range(len(path)-1):
                    self.feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = (self.sigm-(l+1)/L**(l+1)) + self.feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]

    def fit(self,cluster):
        cluster=np.append(cluster,0)
        self.bestSolution = None
        self.generateGraph(cluster)
        cluster = cluster[cluster!=0]
        for i in range(self.max_iter):
            self.initialPopulation(cluster)
            self.cal_fitness_population()
            self.updateFeromone()
            # print(str(i)+":\t"+str(int(self.bestSolution.distance))+"\t")

        self.best_fitness_global += self.bestSolution.fitness
        self.best_distance_global += self.bestSolution.distance
        self.best_route_global.append(self.bestSolution.customerList)
        self.route_count_global += len(self.bestSolution.customerList)
    
    # def fit_allClusters(self, clusters):
    #     bestSolutions = []
    #     for i in range(len(clusters)):
    #         bestSolutions.append(self.fit(cluster=clusters[i]))
    #     return bestSolutions

    def fit_allClusters(self, clusters):
        for i in range(len(clusters)):
            self.fit(cluster=clusters[i])
        return self.best_fitness_global,self.best_route_global,self.best_distance_global,self.route_count_global

if __name__ == "__main__":
    import time
    #Thông số K-means
    N_CLUSTER = 1
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100
    #Thông số ACO
    ALFA = 2
    BETA = 5
    SIGM = 3
    RO = 0.8
    TH = 80
    NUM_ANTS = 22
    ITERATION = 100

    VEHCICLE_CAPACITY = 200
    M = 0
    #Thông số ngoài
    DATA_ID = "R101" #File dữ liệu cụ thể
    DATA_NAME = "R1" #Bộ dữ liệu
    RUN_TIMES = 1
    EXCEL_FILE= None # File excel xuất ra kết quả bộ dữ liệu
    FILE_EXCEL_PATH = "result/"
    #================================================================================================================
    if (DATA_ID != None):
        url_data = "data/csv/"+DATA_ID[:-2]+"/"
        data_files = [DATA_ID+".csv"]
        
    else:
        url_data = "data/csv/"+DATA_NAME+"/"
        data_files = sorted([f for f in os.listdir(url_data) if f.endswith(('.csv'))])
        EXCEL_FILE = FILE_EXCEL_PATH + DATA_NAME + ".xlsx"

    len_data = len(data_files)
    print(f"Bộ dữ liệu {DATA_NAME}: {data_files}")
    run_time_data = 0
    route_count_data = 0
    distance_data = 0
    fitness_data = 0
    C_scope = range(1,10)
    data_excel =[]
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
        # kmeans.elbow_k_means_lib(data_kmeans,C_scope)
        # kmeans.elbow_k_means(data_kmeans,C_scope)
        #-------------------------------------------------------------------------------------------------------------------------------------

        print("#ACO =============================")


        for j in range(RUN_TIMES):
            aco = ACO(num_ants=NUM_ANTS,max_iter=ITERATION,vehcicle_capacity=VEHCICLE_CAPACITY,M=M,alfa=ALFA,sigm=SIGM,beta=BETA,ro=RO,th=TH,customers=customers)
            _start_time = time.time()
            best_fitness_global,best_route_global,best_distance_global,route_count_global= aco.fit_allClusters(clusters=cluster)
            run_time = round_float(time.time() - _start_time)
            run_time_mean += run_time
            print(f"Thời gian chạy {data_file[:-4]} lần",j+1,":", run_time)
            # print("Fitness: ", round_float(best_fitn/ess_global))
            print("Distance: ", round_float(best_distance_global))
            print("Số lượng route: ",route_count_global)
            # print(best_route_global)
            route_count_mean += route_count_global
            distance_mean += best_distance_global
            fitness_mean += best_fitness_global
            data_excel+=[[route_count_global,round_float(best_distance_global),round_float(run_time)]]
            print("===================================")
        #Thống kê file dữ liệu

        print(f"#Thống kê {data_file[:-4]} =============================")
        print("Số lượt chạy mỗi bộ dữ liệu ",RUN_TIMES)
        # print("Fitness trung bình: ", round_float(fitness_mean/RUN_TIMES))
        print("Số lượng route trung bình: ", round_float(route_count_mean/RUN_TIMES))
        print("Thời gian di chuyển trung bình: ", round_float(distance_mean/RUN_TIMES))
        print("Thời gian chạy trung bình: ",round_float(run_time_mean/RUN_TIMES))
        run_time_data += round_float(run_time_mean/RUN_TIMES)
        route_count_data += round_float(route_count_mean/RUN_TIMES)
        distance_data += round_float(distance_mean/RUN_TIMES)
        fitness_data += round_float(fitness_mean/RUN_TIMES)
        data_excel+=[[round_float(route_count_mean/RUN_TIMES),round_float(distance_mean/RUN_TIMES)]]
        print("====================================================================================================================")

    #Thống kê data   
    print("=====================================================================================================================================")
    print(f"#Thống kê {DATA_NAME} =============================")
    print("Số lượt chạy mỗi bộ dữ liệu ",RUN_TIMES)
    # print("Fitness trung bình: ", round_float(fitness_data/len_data))
    print("Số lượng route trung bình: ", round_float(route_count_data/len_data))
    print("Thời gian di chuyển trung bình: ", round_float(distance_data/len_data))
    print("Thời gian chạy trung bình: ",round_float(run_time_data/len_data))
    