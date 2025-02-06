import pandas as pd
import numpy as np
from collections import namedtuple 
import os
# Warehouse = namedtuple('Warehouse',['xy_coord','dueDate'])
Customer = namedtuple('Customer',['id','xy_coord','demand','readyTime','dueTime','serviceTime'])

def load_csv_dataset(url=None,name_of_id=None, number_of_customer: int = None):
  # folder = file_name[0:2]
  # url = "Python\\Genetic_Algorithm\\data\\csv\\" + folder +"\\"+ name_of_id+ ".csv"

  path = os.path.join(url, name_of_id)
  f = pd.read_csv(path)
  print("dataset: ",name_of_id[0:4])
  print("Number of customer: ",number_of_customer)
  # data_kmeans = np.delete(np.array(f[['XCOORD.','YCOORD.']]),0,0)   
  data = np.array(f[['XCOORD.','YCOORD.']])
  # data = np.array(f[['XCOORD.','YCOORD.','DUE DATE']])

  demands = np.array(f['DEMAND'])
  ready_time = np.array(f['READY TIME'])
  due_date =  np.array(f['DUE DATE'])
  service_time =  np.array(f['SERVICE TIME'])

  customers = []
  for i in range(number_of_customer+1):
    # customers.append(Customer(i-1,data[i],demands[i],ready_time[i],due_date[i],service_time[i]))
    customers.append(Customer(i,data[i],demands[i],ready_time[i],due_date[i],service_time[i]))
  # warehouse = Warehouse(data[0],due_date[0])
  
  # return data,customers,warehouse
  return data[:number_of_customer+1],customers

# print(load_csv_dataset(url="data/csv/R1/",name_of_id="R102.csv",number_of_customer=25))

def load_txt_dataset(url=None,name_of_id=None):
  # folder = file_name[0:2]
  # url = "Python\\Genetic_Algorithm\\data\\csv\\" + folder +"\\"+ name_of_id+ ".csv"

  path = os.path.join(url, name_of_id)
  with open(path, 'r') as file:
    lines = file.readlines()
  customers = []
  cord_data = []
  for idx,line in enumerate(lines[9:]):  # Skip the header line
      data = line.strip().split()
      xcoord = np.int64(data[1])
      ycoord = np.int64(data[2])
      demand = np.int64(data[3])
      ready_time = np.int64(data[4])
      due_date = np.int64(data[5])
      service_time = np.int64(data[6])

      cord_data.append([xcoord,ycoord])
      customers.append(Customer(idx,np.array([xcoord,ycoord]), demand, ready_time, due_date, service_time))

  
  # return data,customers,warehouse
  return np.array(cord_data),customers

# print(load_txt_dataset(url="data/txt/100/R1/",name_of_id="R101.txt"))

