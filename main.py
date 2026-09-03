import numpy as np
from algorithm.GeneticAlgorithm import GA
from algorithm.GACombineVNS import GA_VNS
from algorithm.FuzzyCMedoids import FuzzyCMedoids
from algorithm.DiscretePSO import DiscretePSO
from algorithm.FSRD_SP import FSRD_SP
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
from ultility.utilities import (
    round_float, write_excel_file,
    visualize_vrptw_clusters, visualize_fuzzy_clusters,
    plot_comparison_results,
    create_graph,
)
import signal
import logging
import os
import time
import sys

if __name__ == "__main__":
    _start_time_all = time.time()
    # ===============================Thiết lập thuật toán chạy===============================
    # Danh sách các thuật toán định tuyến: 'FSRD_SP', 'GA_VNS', 'GA', 'DiscretePSO'
    algorithms = ['FSRD_SP']
    # Danh sách các thuật toán phân cụm: 'FCM' (hoặc 'FuzzyCMedoids'), 'KMeans'
    clustering_algorithms = ['FCM']
    data_names = ['R1']
    # data_names = ['R1','R2','C1','C2','RC1','RC2']
    # ===============================Thiết lập các thông số===============================
    # Thông số VRPTW
    M = 0  # Sai số trong cửa sổ thời gian
    N_CLUSTER = 20  # Số lượng cụm
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100  # Số lượng khách hàng
    # Thông số Fuzzy c-Medoids
    FCM_KAPPA = 2.0        # Hệ số mờ (>= 2)
    FCM_EPSILON = 1e-4     # Ngưỡng hội tụ
    FCM_MAX_ITER = 300     # Số vòng lặp tối đa
    FCM_RHO = 0.3          # Ngưỡng khách hàng biên mờ
    # Thông số chạy GA
    INDIVIDUAL = 150  # Số lượng cá thể
    GENERATION = 150  # Số lượng đời quần thể
    CROSSOVER_RATE = 0.8  # Tỉ lệ trao đổi chéo
    MUTATION_RATE = 0.15  # Tỉ lệ đột biến
    CONSERVE_RATE = 0.1  # Tỉ lệ bảo tồn
    # Thông số VNS
    LIST_N_L = [2] * 6
    BETA_0 = 5
    BETA_1 = 2
    # Thông số Discrete PSO
    PSO_PARTICLES = 30     # Số lượng hạt
    PSO_MAX_ITER = 100     # Số vòng lặp tối đa
    PSO_W = 0.8            # Trọng số quán tính
    PSO_C1 = 0.5           # Hệ số học cá nhân
    PSO_C2 = 0.5           # Hệ số học xã hội
    PSO_SDITER = 50        # Dissolution Rule: số thế hệ không cải thiện
    # Thông số bộ dữ liệu chạy
    TITLE_NAMES = ['Route', 'Distance', 'Fitness', 'RunTime']
    DATA_ID = None  # File dữ liệu cụ thể
    DATA_NUMBER_CUS = "200"  # Số lượng khách hàng
    RUN_TIMES = 1  # Số lượng chạy
    FILE_EXCEL_PATH = "result/"
    # Hiển thị hình ảnh phân cụm (True = blocking cho đến khi đóng cửa sổ)
    VISUALIZE_CLUSTERS = False
    # Tự động vẽ biểu đồ so sánh các thuật toán sau khi chạy xong (True = hiển thị và lưu ảnh)
    PLOT_COMPARISON = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("terminal_output.log", mode="w", encoding="utf-8"),  # ghi đè file, UTF-8
            logging.StreamHandler(sys.stdout)  # in ra console
        ]
    )

    for data_name in data_names:
        DATA_NAME = data_name

        for clustering_algorithm in clustering_algorithms:
            # Nhãn thuật toán phân cụm dùng trong tên file và worksheet Excel
            CLUSTER_LABEL = "FCM" if clustering_algorithm in ["FCM", "FuzzyCMedoids"] else clustering_algorithm
            alg_str = '+'.join(algorithms)
            FILE_NAME = f"_{alg_str}_{CLUSTER_LABEL}"

            if DATA_ID is not None:
                url_data = "data/txt/" + DATA_NUMBER_CUS + "/" + DATA_ID[:-2] + "/"
                data_files = [DATA_ID + ".txt"]
                EXCEL_FILE = FILE_EXCEL_PATH + DATA_ID + FILE_NAME + ".xlsx"
            else:
                url_data = "data/txt/" + DATA_NUMBER_CUS + "/" + DATA_NAME + "/"
                data_files = sorted(
                    [f for f in os.listdir(url_data) if f.endswith('.txt')]
                )
                EXCEL_FILE = FILE_EXCEL_PATH + DATA_NAME + FILE_NAME + ".xlsx"

            # ===============================Thiết lập xử lý tín hiệu dừng===============================
            def signal_handler(sig, frame):
                """Hàm xử lý khi nhận tín hiệu dừng chương trình"""
                print("Chương trình bị dừng, lưu dữ liệu...")
                if DATA_NAME is not None:
                    write_excel_file(
                        data_excels=data_excels,
                        data_files=data_files,
                        data_name=DATA_NAME,
                        run_time=RUN_TIMES,
                        algorithms=algorithms,
                        title_names=TITLE_NAMES,
                        fileio=EXCEL_FILE,
                        cluster_label=CLUSTER_LABEL,
                    )
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            data_excels = np.zeros((len(algorithms), len(data_files), RUN_TIMES, len(TITLE_NAMES)))
            try:
                # Thiết lập chạy các thuật toán
                for idx_al, algorithm in enumerate(algorithms):
                    len_data = len(data_files)
                    logging.info(f"Bộ dữ liệu {DATA_NAME} [Phân cụm: {clustering_algorithm}] [Định tuyến: {algorithm}]: {data_files}")
                    run_time_data = 0
                    route_count_data = 0
                    distance_data = 0
                    fitness_data = 0
                    for idx_dat, data_file in enumerate(data_files):
                        run_time_mean = 0
                        route_count_mean = 0
                        distance_mean = 0
                        fitness_mean = 0
                        _start_time = time.time()
                        # Khởi tạo dữ liệu
                        vehicle_capacity, cord_data, customers = load_txt_dataset(
                            url=url_data, name_of_id=data_file
                        )

                        graph_data = create_graph(coords=cord_data)

                        VEHICLE_CAPACITY = vehicle_capacity
                        logging.info(f"Trọng tải tối đa: {VEHICLE_CAPACITY}")
                        lower_bound = np.ceil(np.sum(customers[:, 3]) / VEHICLE_CAPACITY).astype(int)
                        logging.info(f"Số xe tối thiểu sử dụng: {lower_bound}")
                        n_cluster_current = min(N_CLUSTER, lower_bound)
                        if n_cluster_current != N_CLUSTER:
                            logging.info(f"Số cụm được điều chỉnh thành: {n_cluster_current}")

                        logging.info(f"Thời gian lấy dữ liệu: {round_float(time.time() - _start_time)}")

                        warehouse = cord_data[0]
                        data_kmeans = np.delete(cord_data, 0, 0)

                        is_fsrd = algorithm == "FSRD_SP"

                        # ── Phân cụm theo thuật toán đã chọn ──────────────────────────────
                        match clustering_algorithm:
                            case "FCM" | "FuzzyCMedoids":
                                logging.info("#Fuzzy c-Medoids =============================")
                                fcm = FuzzyCMedoids(
                                    n_cluster=n_cluster_current,
                                    vehicle_capacity=VEHICLE_CAPACITY,
                                    kappa=FCM_KAPPA,
                                    epsilon=FCM_EPSILON,
                                    max_iter=FCM_MAX_ITER,
                                    rho=FCM_RHO,
                                )
                                fcm.fit(customers)
                                if is_fsrd:
                                    clusters = fcm.get_overlapping_subproblems(rho=FCM_RHO)
                                    logging.info(
                                        "FSRD-SP Overlapping Subproblems (rho=%s): %s",
                                        FCM_RHO,
                                        [len(c) for c in clusters],
                                    )
                                else:
                                    clusters = fcm.get_clusters()

                                boundary = fcm.get_fuzzy_boundary_customers()
                                logging.info(
                                    f"Fuzzy c-Medoids: {fcm.n_iter} vòng lặp, "
                                    f"thời gian = {round_float(fcm.process_time)}s"
                                )
                                logging.info(f"Medoids (1-indexed): {fcm.medoids}")
                                logging.info(f"Khách hàng biên mờ (rho <= {FCM_RHO}): {boundary}")

                                if VISUALIZE_CLUSTERS:
                                    try:
                                        visualize_fuzzy_clusters(
                                            cord_data,
                                            U=fcm.U,
                                            labels=fcm.labels,
                                            medoids=fcm.medoids,
                                            kappa=fcm.kappa,
                                            n_iter=fcm.n_iter,
                                            title_suffix=data_file[:-4],
                                            show=True,
                                        )
                                    except Exception as _ve:
                                        logging.warning(f"Không thể hiển thị ảnh phân cụm: {_ve}")

                            case "KMeans" | "Kmeans":
                                logging.info("#K-means =============================")
                                kmeans = Kmeans(
                                    epsilon=EPSILON,
                                    maxiter=MAX_ITER,
                                    n_cluster=n_cluster_current,
                                )
                                U1, V1, step = kmeans.k_means_lib_sorted(data_kmeans, warehouse)
                                clusters = kmeans.data_to_cluster(U1)

                                if VISUALIZE_CLUSTERS:
                                    try:
                                        visualize_vrptw_clusters(cord_data, clusters)
                                    except Exception as _ve:
                                        logging.warning(f"Không thể hiển thị ảnh phân cụm: {_ve}")

                            case _:
                                raise ValueError(f"Thuật toán phân cụm không được hỗ trợ: {clustering_algorithm}")

                        logging.info(clusters)

                        # ── Định tuyến (Routing) ───────────────────────────────────────────
                        for run in range(RUN_TIMES):
                            match algorithm:
                                case "FSRD_SP":
                                    al = FSRD_SP(
                                        individual=INDIVIDUAL,
                                        generation=GENERATION,
                                        crossover_rate=CROSSOVER_RATE,
                                        mutation_rate=MUTATION_RATE,
                                        vehicle_capacity=VEHICLE_CAPACITY,
                                        conserve_rate=CONSERVE_RATE,
                                        M=M,
                                        customers=customers,
                                        graph_data=graph_data,
                                    )

                                case "GA":
                                    al = GA(
                                        individual=INDIVIDUAL,
                                        generation=GENERATION,
                                        crossover_rate=CROSSOVER_RATE,
                                        mutation_rate=MUTATION_RATE,
                                        vehicle_capacity=VEHICLE_CAPACITY,
                                        conserve_rate=CONSERVE_RATE,
                                        M=M,
                                        customers=customers,
                                        graph_data=graph_data,
                                    )

                                case "GA_VNS":
                                    al = GA_VNS(
                                        individual=INDIVIDUAL,
                                        generation=GENERATION,
                                        crossover_rate=CROSSOVER_RATE,
                                        mutation_rate=MUTATION_RATE,
                                        vehicle_capacity=VEHICLE_CAPACITY,
                                        conserve_rate=CONSERVE_RATE,
                                        M=M,
                                        customers=customers,
                                        graph_data=graph_data,
                                        list_n_l=LIST_N_L,
                                        beta_0=BETA_0,
                                        beta_1=BETA_1,
                                    )

                                case "DiscretePSO":
                                    al = DiscretePSO(
                                        num_particles=PSO_PARTICLES,
                                        max_iter=PSO_MAX_ITER,
                                        vehicle_capacity=VEHICLE_CAPACITY,
                                        M=M,
                                        w=PSO_W,
                                        c1=PSO_C1,
                                        c2=PSO_C2,
                                        customers=customers,
                                        graph_data=graph_data,
                                        sditer=PSO_SDITER,
                                    )

                            logging.info(f"#{algorithm} =============================")

                            best_fitness_global, best_route_global, best_distance_global, route_count_global, process_time = al.fit(clusters=clusters)

                            run_time_mean += process_time

                            logging.info(
                                f"Thời gian chạy {data_file[:-4]} lần {run+1}: {round_float(process_time)}"
                            )
                            logging.info(f"Fitness: {round_float(best_fitness_global)}")
                            logging.info(f"Distance: {round_float(best_distance_global)}")
                            logging.info(f"Số lượng route: {route_count_global}")
                            logging.info(best_route_global)
                            route_count_mean += route_count_global
                            distance_mean += best_distance_global
                            fitness_mean += best_fitness_global
                            data_excels[idx_al][idx_dat][run] = np.array([
                                route_count_global,
                                round_float(best_distance_global),
                                round_float(best_fitness_global),
                                round_float(process_time),
                            ])
                            logging.info("===================================")

                        # Thống kê file dữ liệu
                        logging.info(f"#Thống kê {data_file[:-4]} =============================")
                        logging.info(f"Số lượt chạy mỗi bộ dữ liệu: {RUN_TIMES}")
                        logging.info(f"Fitness trung bình: {round_float(fitness_mean/RUN_TIMES)}")
                        logging.info(f"Số lượng route trung bình: {round_float(route_count_mean/RUN_TIMES)}")
                        logging.info(f"Thời gian di chuyển trung bình: {round_float(distance_mean/RUN_TIMES)}")
                        logging.info(f"Thời gian chạy trung bình: {round_float(run_time_mean/RUN_TIMES)}")
                        logging.info("====================================================================================================================")

                        route_count_data += route_count_mean / RUN_TIMES
                        distance_data += distance_mean / RUN_TIMES
                        fitness_data += fitness_mean / RUN_TIMES
                        run_time_data += run_time_mean / RUN_TIMES

                    # Thống kê data
                    logging.info("=====================================================================================================================================")
                    logging.info(f"#Thống kê {DATA_NAME} [Phân cụm: {clustering_algorithm}] =============================")
                    logging.info(f"Số lượt chạy mỗi bộ dữ liệu: {RUN_TIMES}")
                    logging.info(f"Fitness trung bình: {round_float(fitness_data/len_data)}")
                    logging.info(f"Số lượng route trung bình: {round_float(route_count_data/len_data)}")
                    logging.info(f"Thời gian di chuyển trung bình: {round_float(distance_data/len_data)}")
                    logging.info(f"Thời gian chạy trung bình: {round_float(run_time_data/len_data)}")
                    logging.info(f"HOÀN THÀNH THUẬT TOÁN, thời gian chạy toàn bộ: {round_float(time.time() - _start_time_all)}")

            except Exception as e:
                logging.exception("Error: %s", e)

            finally:
                if DATA_NAME is not None:
                    write_excel_file(
                        data_excels=data_excels,
                        data_files=data_files,
                        data_name=DATA_NAME,
                        run_time=RUN_TIMES,
                        algorithms=algorithms,
                        title_names=TITLE_NAMES,
                        fileio=EXCEL_FILE,
                        cluster_label=CLUSTER_LABEL,
                    )

                    if PLOT_COMPARISON:
                        try:
                            chart_save_path = FILE_EXCEL_PATH + DATA_NAME + FILE_NAME + "_chart.png"
                            plot_comparison_results(
                                data_excels=data_excels,
                                data_files=data_files,
                                algorithms=algorithms,
                                title_names=TITLE_NAMES,
                                data_name=DATA_NAME,
                                cluster_label=CLUSTER_LABEL,
                                save_path=chart_save_path,
                                show=True,
                            )
                        except Exception as _pe:
                            logging.warning(f"Không thể vẽ biểu đồ so sánh: {_pe}")

