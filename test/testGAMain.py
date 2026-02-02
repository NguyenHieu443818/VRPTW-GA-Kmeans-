from algorithm.GeneticAlgorithm import GA
from algorithm.kmeans import Kmeans
import numpy as np
from ultility.readDataFile import load_txt_dataset
from ultility.utilities import round_float, write_excel_file, create_graph


if __name__ == "__main__":
    import time
    _start_time = time.time()
    n_cluster = 6
    epsilon = 1e-5
    MAX_ITER = 1000
    # #Khởi tạo dữ liệu
    # data, customers = load_csv_dataset(
    #     url="data/csv/R1/", name_of_id="R101.csv", number_of_customer=100)

    vehicle_capacity, cord_data, customers = load_txt_dataset(
        url="data/txt/200/R2/", name_of_id="R2_2_1.txt")

    # #chạy kmeans
    # kmeans = Kmeans(epsilon=epsilon, maxiter=MAX_ITER)
    # data_kmeans = np.delete(data,0,0)
    # U1, V1, step = kmeans.k_means(data_kmeans, n_cluster)

    # cluster = [np.argwhere([U1==i]).T[1,] for i in range(n_cluster]
    # customer = [customers[list(cluster[1])[i]]for i in range(len(cluster[1])]

    # print('customer',customer)
    # n = int(input("Nhập id1 "))
    # x = int(input("Nhập id2 "))

    # customer = customers[n]
    # print(np.linalg.norm(customers[n].xy_coord-customers[x].xy_coord))

    # for i in range(len(cluster)):
    #     #chạy GA
    #     GA = GeneticAlgorithm(individual=10)
    #     population = GA.initial_population(cluster[i])
    #     print(GA.individual_to_route(population[1, customers,warehouse))
    #     for j in range(GA.__generation):

    #         pass
    #     bestIndividual = None
    graph = create_graph(coords=cord_data)
    GA = GA(individual=10, M=0, customers=customers, graph_data=graph)
    # GA.initial_population(cluster[1])
    # print('cluster[1] ',cluster[1])
    # print('population[1] ',population[1])
    # print(GA.individual_to_route(population[1, customers))
    # print(GA.individual_to_route([2, 21, 73, 41, 56, 4, 52, 6, customers))

    test = [2, 21, 73, 41, 56, 4, 5, 83, 61, 85, 37, 93, 14, 44, 38, 43, 13, 27, 69, 76, 79, 3, 54, 24, 80, 28, 12, 40, 53, 26, 30, 51, 9, 66, 1, 31, 88, 7, 10, 33, 29, 78, 34, 35, 77, 36, 47, 19, 8, 46,
            17, 39, 23, 67, 55, 25, 45, 82, 18, 84, 60, 89, 52, 6, 59, 99, 94, 96, 62, 11, 90, 20, 32, 70, 63, 64, 49, 48, 65, 71, 81, 50, 68, 72, 75, 22, 74, 58, 92, 42, 15, 87, 57, 97, 95, 98, 16, 86, 91, 100]
    a = [19, 96, 40, 50, 69, 17, 65, 7, 93, 44, 46, 8, 33, 36, 81, 30, 10, 48, 47, 51, 27, 3, 1, 18, 57, 22, 87, 75, 14, 5, 73, 15, 74, 21, 52, 28, 84, 86, 60, 91, 12, 76, 55, 89, 31, 61,
         85, 94, 92, 58, 63, 64, 98, 23, 53, 24, 25, 82, 78, 71, 39, 59, 6, 29, 77, 2, 79, 37, 100, 97, 9, 88, 80, 26, 83, 11, 45, 32, 38, 95, 20, 70, 99, 72, 68, 90, 62, 41, 49, 54, 16, 4]
    a= [2, 15, 87, 57, 40, 53, 13, 58,59, 95, 98, 61, 85, 94, 97, 37, 91, 93,92, 42, 99, 6, 96, 100,14, 44, 16, 86, 38, 43,45, 83, 82, 18, 8, 84, 17, 60, 89,5, 52,11, 64, 49, 46, 48,36, 47, 19, 7,31, 30, 90, 10, 32, 70,63, 62, 88,33, 71, 81, 78, 34, 35,65, 51, 9, 66, 20,27, 69, 76, 79, 3, 50, 68, 77, 1,28, 12, 29, 26, 24, 80,39, 67, 54, 55, 25, 4,72, 21, 73, 75, 22, 41, 56, 74,23]
    
    # print(GA.cal_fitness_individualV2(individual=a,customers=customers,graph_data=graph,vehicle_capacity=vehicle_capacity,M=0,fitness_to_branch_bound=None))
    # print(GA.individual_to_route(individual=a))
    a = [39, 69, 28, 27, 29, 67, 12, 76, 79, 3, 50, 55, 26, 54, 1, 68, 24, 4, 25, 80, 77]
    b = [79, 12, 80, 24, 39, 76, 28, 1, 26, 55, 25, 68, 3, 4, 50, 69, 29, 67, 27, 54, 77]
    print(GA.cal_fitness_individualV2(individual=a,customers=customers,graph_data=graph,vehicle_capacity=vehicle_capacity,M=0))
    print(GA.cal_fitness_individualV2(individual=b,customers=customers,graph_data=graph,vehicle_capacity=vehicle_capacity,M=0))

    
    
    
    
