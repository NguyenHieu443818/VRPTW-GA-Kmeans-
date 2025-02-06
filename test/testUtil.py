# # import random
# # import time
from abc import ABC, abstractmethod
import numpy as np
import random
# import math
# # # dad =[54, 27, 24, 79, 28, 67, 53, 38, 66, 25,  3, 23, 11]
# # # mom = [11, 53, 27, 66, 67, 38, 25, 24,  3, 28, 23, 79, 54]
# # dad = [1,2,3,5,8,4,7,6,9]
# # mom = [5,9,3,8,1,2,6,4,7]

# # mom_i = set(mom[3:5+1])
# # # Lọc các phần tử của arr1 theo mặt nạ
# # result = [x for x in dad if x not in mom_i]

# # print(result)

# # child = [result.pop(0) for _ in range(0,3] + mom[3:5+1] + [result.pop(0) for _ in range(6,len(mom)]


# # print(mom[9:12+1])
# # dad_filter = [dad!=mom[9:12+1] for i in mom]
# # print(dad_filter)

# # child=[0]*len(mom)
# # child =[dad[i] for i in range(len(dad)) if i not in range(3,5+1) and not any(mom[t] == dad[i] for t in range(3,5+1)]

# # child += mom[9:12+1]


# # vt = 0
# # i=0
# # while  vt < len(mom) :
# #     if vt not in range(9,12+1):
# #         if not any([True for t in range(9, 12 + 1) if mom[t] == dad[i]):
# #             child.append(dad[i])
# #             vt+=1
# #         i+=1
# #     else:
# #         child += mom[9:12+1]
# #         vt+=(12+1-9)

# # print(child)
# # child = dad[] + mom[9,12+1] + dad[:]


# # a = [21, 19, 23, 18, 20, 22, 24, 25,26, 27, 28, 29, 32, 33, 34, 30,31, 48, 49, 50,51, 52, 56, 57, 58,64, 65, 66, 67, 59,62, 71, 76, 77, 80,83, 63, 84, 85, 74, 86,75, 87, 89, 91,92, 95, 97, 99, 93, 94,5, 1, 2, 3, 4, 6, 7, 8,9, 10, 11, 12, 15, 16, 17,35, 36, 37, 38, 39, 40, 41, 42, 43, 44,45, 46, 47, 53, 54, 96,98, 60, 55, 61, 81, 90, 68, 70,72, 69, 73, 78, 79, 100,82, 13, 14, 88]
# # print(len(a))

# file_path = 'input.txt'
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # Initialize the customers dictionary
# customers = {}
# num_customers = 100
# end_point_data = num_customers + 2
# cord_data = []
# cust_index = 0
# # Process each line and create the dictionary entries
# for line in lines[1:end_point_data]:  # Skip the header line
#     list_cord = []
#     data = line.strip().split()
#     cust_no = cust_index
#     xcoord = float(data[1])
#     ycoord = float(data[2])
#     demand = int(data[3])
#     ready_time = int(data[4])
#     due_date = int(data[5])
#     service_time = int(data[6])
#     list_cord.append(xcoord)
#     list_cord.append(ycoord)
#     cord_data.append(list_cord)
#     customers[cust_no] = (cust_no,xcoord, ycoord, demand, ready_time, due_date, service_time)
#     cust_index += 1

# def calculate_distance(coord1, coord2):
#     x_diff = coord1[1] - coord2[1]
#     y_diff = coord1[2] - coord2[2]
#     distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
#     return distance

# def split_route(route):
#     total_demand = 0
#     demand_groups = []
#     current_group = []
#     service_time = 0
#     cus_before = 0
#     for i in route:
#         print(service_time)
#         if service_time != 0:
#             service_time = service_time + customers[cus_before][6] + calculate_distance(customers[i],customers[cus_before])
#         if service_time == 0:
#             service_time = customers[i][4]
#         if total_demand + customers[i][3] <= 200 and (service_time >= (customers[i][4] - 0) and service_time <= (customers[i][5] + 0)):
#             current_group.append(i)
#             total_demand += customers[i][3]
#         else:
#             demand_groups.append(current_group)
#             current_group = [i]
#             total_demand = customers[i][3]
#             service_time = 0
#         cus_before = i
#     demand_groups.append(current_group)
#     return demand_groups

# print(split_route([5, 45, 2, 7, 6, 8, 3, 1, 70, 100,14,47]))
# a = 'RC101.csv'
# print(a[:-4])
# b = ['Lần chạy'] + (['Route','Distance']*3)
# print(b)

# a = [[2,3]
# b= a+[[4,5],6,7],8,9],10,11],10,11]
# print(b)
# b = np.array(b)
# b=b.reshape(2,3,2) # số bộ dữ liệu x số lần chạy x số thuộc tính
# result = b.swapaxes(0, 1).reshape(3, 4) # số lần chạy x (số bộ dữ liệu x số thuộc tính)
# result2 = b.transpose(1,0,2).reshape(3,4)
# print(b)
# print(result)
# print(result2)
# a = [1]
# a = a+[[4],6],8],10],111]
# print(a)
# a = np.random.uniform(-5, 5, size=(1000,102))
# print((a[:,:-2]).shape)
# arr_2d = np.array([[1, 2, 3, 4],
#                    [5, 6, 7, 8],
#                    [9, 10, 11, 12])
# print((arr_2d[:,-2:]))
# a = ((12*4+1) // 26)*"A" + chr(ord('A')+((12*4+1)% 26))
# print(a)

# a = [77, 47, 32, 71, 54, 96, 92, 60, 87, 88, 71, 94, 81, 11, 78, 69, 39, 47, 87, 53, 38, 90, 89, 22]
# a = [38, 6, 85, 14, 93 ,59, 42, 14, 92, 94, 59, 96, 91, 96,42, 14, 44, 87, 94, 94, 85, 13, 96, 99, 12, 29, 55, 4, 68, 80, 28, 39, 67, 24, 25,39, 26,30, 30, 33, 1, 70,65, 34, 9, 35,65, 51, 71, 71, 20,65, 78, 81, 78, 3, 77,7, 10, 32,36, 36, 19, 11, 90, 48, 48,36, 64, 63, 88, 19,27, 52, 5, 83, 60, 60,45, 45, 18, 8, 84, 17, 89,2, 2, 21, 22, 56, 56, 41, 41, 58,72, 73, 23, 22, 74, 74]

# a = np.array(a)
# print(np.sort(np.unique(a)))

# a = [42,45,1,3,42] # child
# b = [5,4,1,2,3] # dad
# a_cop = a.copy()
# b_cop = b.copy()

# sort_a = np.sort(np.array(a))
# sort_b = np.sort(np.array(b))
# for i in range(len(a)):
#     a_cop[a_cop.index(sort_a[i])] = i

# a_cop = [sort_b[i] for i in a_cop]
# # print(a_cop)
# print(a_cop)

# a= [np.int64(60),np.int64(15), np.int64(14), np.int64(85), np.int64(43), np.int64(97), np.int64(5),np.int64(57), np.int64(16), np.int64(45), np.int64(99),np.int64(93), np.int64(6),np.int64(59), np.int64(61), np.int64(38), np.int64(17), np.int64(94), np.int64(96),np.int64(98), np.int64(89), np.int64(83),np.int64(44), np.int64(37), np.int64(13), np.int64(84),np.int64(87), np.int64(91), np.int64(95), np.int64(92),np.int64(100),np.int64(42), np.int64(86),np.int64(25), np.int64(55),np.int64(40), np.int64(28), np.int64(72), np.int64(4), np.int64(12),np.int64(54), np.int64(39), np.int64(75),np.int64(22), np.int64(24), np.int64(21),np.int64(74), np.int64(23),np.int64(41), np.int64(58),np.int64(2), np.int64(73), np.int64(67), np.int64(26), np.int64(53),np.int64(56),np.int64(29), np.int64(78), np.int64(69), np.int64(68),np.int64(34), np.int64(65), np.int64(51),np.int64(3), np.int64(79), np.int64(70),np.int64(66), np.int64(9), np.int64(1),np.int64(20), np.int64(76), np.int64(80),np.int64(71), np.int64(35), np.int64(77),np.int64(81), np.int64(27), np.int64(33), np.int64(50),np.int64(30),np.int64(52), np.int64(62), np.int64(88), np.int64(19), np.int64(10), np.int64(8),np.int64(47), np.int64(49), np.int64(90), np.int64(11),np.int64(36), np.int64(64), np.int64(18), np.int64(82), np.int64(31), np.int64(7),np.int64(46), np.int64(32), np.int64(63),np.int64(48)]
# print(np.unique(np.array(a)).shape)

# lai=0
# khachhang = "moi"
# slm = 9
# tien = 0
# # Kiểm tra lãi theo khách hàng
# if(khachhang=="moi"):
#     lai+=0.15
# else:
#     lai += 0.1

# # Kiểm tra lãi theo số lượng
# if(slm>10):
#     lai+=0.1
# elif(slm>=6): # Chắc chắn <=10
#     lai+=0.05

# # Kiếm tra đơn hàng
# if(tien>50):
#     lai+= 0.15
# elif(tien>=10):
#     lai += 0.08
# # Ưu đãi khách hàng mới
# if(khachhang=="moi" and slm>10 and tien >100):
#     lai += 0.2


# tien *= (1-lai)

# # nhập 1 số nguyên dương
# while (a<=0):
#     a = int(input("Nhap so a"))

# d1,d2,d3 = 3,1,2

# if d1>d2:
#     if d2>d3:
#         print("chạy 1")
#         # d1 lớn nhất
#         # d3 bé nhất
#     elif d3>d1: # d3<=d2
#         print("chạy 2")
#         #d3 lớn nhất
#         #d2 bé nhất
#     else: # d3<d1
#         print("chạy 3")
#         # d1 lớn nhất
#         # d2 bé nhất
# else: # d2>d1
#     if d2<d3:
#         print("chạy 4")
#         # d3 lớn nhất
#         # d1 bé nhất
#     elif d3<d1: # d3>d2
#         print("chạy 5")
#         #d3 lớn nhất
#         #d1 bé nhất
#     else: # d3>d1
#         print("chạy 6")
#         # d2 lớn nhất
#         # d1 bé nhất

# array = [[np.int64(59), np.int64(95), np.int64(87), np.int64(94), np.int64(6), np.int64(37), np.int64(97), np.int64(100),np.int64(42), np.int64(38), np.int64(16), np.int64(43), np.int64(13),np.int64(14), np.int64(98), np.int64(99), np.int64(86), np.int64(91), np.int64(93),np.int64(92), np.int64(61), np.int64(44), np.int64(85), np.int64(96),np.int64(28), np.int64(12), np.int64(26), np.int64(24), np.int64(80),np.int64(29), np.int64(54), np.int64(68),np.int64(39), np.int64(67), np.int64(55), np.int64(4), np.int64(25),np.int64(76), np.int64(78), np.int64(34), np.int64(79), np.int64(1),np.int64(65), np.int64(9), np.int64(81), np.int64(50), np.int64(77),np.int64(33), np.int64(69), np.int64(51), np.int64(66), np.int64(20), np.int64(3),np.int64(30), np.int64(71), np.int64(35), np.int64(70),np.int64(31), np.int64(62), np.int64(88), np.int64(7), np.int64(10), np.int64(32),np.int64(36), np.int64(47), np.int64(64), np.int64(49), np.int64(48),np.int64(63), np.int64(11), np.int64(19), np.int64(90),np.int64(27), np.int64(52), np.int64(5), np.int64(84), np.int64(46), np.int64(60), np.int64(89),np.int64(45), np.int64(83), np.int64(82), np.int64(18), np.int64(8), np.int64(17),np.int64(72), np.int64(2), np.int64(21), np.int64(75), np.int64(22), np.int64(41), np.int64(56), np.int64(74),np.int64(23), np.int64(73), np.int64(57), np.int64(58),np.int64(15), np.int64(40), np.int64(53)]]
# re = np.array(array).flatten()
# # print(re)

# dad = [1,2,3,4,5,6,7,8]
# mom = [8,7,3,2,4,1,5,6]
# pos1 = 2
# pos2 = 3
# gene_child=[0]*len(mom)
# pos1_mom = mom.index(dad[pos1])
# pos2_mom = mom.index(dad[pos2])


# if(dad[pos1:pos2+1] == mom[pos1_mom:pos2_mom+1]):
#     print("Có vào")
#     pos2_mom += random.randint(0, len(mom)-len(dad[pos1:pos2+1]))

# print(len(mom)-len(dad[pos1:pos2+1]))
# print(pos2_mom)
# gene_child[pos1:pos2+1] = dad[pos1:pos2+1]
# filter_mom =  [mom[(x+pos2_mom)%len(mom)] for x in range(len(mom)) if mom[(x+pos2_mom)%len(mom)] not in gene_child]
# print(filter_mom)
# gene_child[pos2+1:] = filter_mom[:len(gene_child[pos2+1:])]
# gene_child[:pos1] = filter_mom[len(gene_child[pos2+1:]):]
# print(gene_child)
dad = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
mom = np.array([5, 4, 6, 9, 2, 1, 7, 8, 3])

pos1 = 2
pos2 = 6
cut_dad = dad[pos1:pos2]
cut_mom = mom[pos1:pos2]

# gene_child_1 = np.zeros(len(mom),dtype=int)
# gene_child_2 = np.zeros(len(dad),dtype=int)
# # Ghép đoạn gen của bố mẹ cho con

# # Lập ma trận ánh xạ bố-mẹ
# mapping_gene= np.vstack((dad[pos1:pos2],mom[pos1:pos2]))

# # Thiết lập ánh xạ
# for idx_p,(d,m) in enumerate(zip(dad,mom)):
#     if idx_p in range(pos1,pos2):
#         gene_child_1[idx_p] = mom[idx_p]
#         gene_child_2[idx_p] = dad[idx_p]
#         continue

#     idx = idx_p
#     # Ánh xạ phần gen của bố
#     i = 1
#     while d in mapping_gene[i] :
#         idx = np.argwhere(mapping_gene[i]==d)[0][0]
#         d = mapping_gene[np.abs(i-1)][idx]
#     # Ánh xạ phần gen của mẹ
#     i = 0
#     while m in mapping_gene[i] :
#         idx = np.argwhere(mapping_gene[i]==m)[0][0]
#         m = mapping_gene[np.abs(i-1)][idx]

#     gene_child_1[idx_p] = d
#     gene_child_2[idx_p] = m


# print(np.flip(dad))

# print(next, type(next))


# import random

# # Mảng 2 chiều
# array_2d = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]

# # Chọn ngẫu nhiên một hàng trong mảng 2 chiều
# random_row = random.choice(array_2d)



# # Chọn ngẫu nhiên một phần tử trong hàng đã chọn
# random_element = random.choice(random_row)

# print("Mảng 2 chiều:")
# for row in array_2d:
#     print(row)

# print("\nPhần tử ngẫu nhiên được chọn:", random_element)


result = None

try:
    # Tính toán và lưu dữ liệu vào biến result
    result = 10 / 2
    
    # Tạo một lỗi để minh họa
    a = 1 / 0  # Phép chia cho 0 sẽ gây lỗi
    
except Exception as e:
    print(f"Lỗi: {str(e)}")


# Kiểm tra và sử dụng dữ liệu tính toán sau khi xử lý exception
if result is not None:
    print("Kết quả tính toán:", result)