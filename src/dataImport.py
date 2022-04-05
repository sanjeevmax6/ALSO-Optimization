import pandas as pd

# This function imports all data from assets folder and stores it in pandas variable
def importData():
    dataset_path = "./assets/"
    file_list = ['google_1min.csv', 'google_5min.csv', 'google_10min.csv', 'google_20min.csv', 'google_30min.csv', 'google_40min.csv', 'google_50min.csv', 'google_60min.csv']
    data = []
    # Extracting Data
    for file_name in file_list:
        data.append(pd.read_csv(dataset_path+file_name))
    
    # Deleting initial serial number column
    for datasets in data:
        del datasets['SERIAL']

    cpu_data = []
    memory_data = []
    disk_data = []
    
    #Splitting data into CPU, Memory and Disk data
    for datasets in data:
        cpu_data.append(datasets['CPU'])
        memory_data.append(datasets['MEMORY'])
        disk_data.append(datasets['DISK'])
        
    cpu_data_min = []
    cpu_data_max = []

    memory_data_min = []
    memory_data_max = []

    disk_data_min = []
    disk_data_max = []

    # Storing minimum values of CPU, Memory, and Disk data
    for cpu in cpu_data:
        cpu_data_min.append(cpu.min())
        cpu_data_max.append(cpu.max())

    for mem in memory_data:
        memory_data_min.append(mem.min())
        memory_data_max.append(mem.max())

    for disk in disk_data:
        disk_data_min.append(disk.min())
        disk_data_max.append(disk.max())
    
    # Standardizing data to lie between 0 and 1
    i=0
    for cpu in cpu_data:
        cpu -= cpu_data_min[i]
        cpu /= cpu_data_max[i]
        i+=1

    i=0
    for mem in memory_data:
        mem -= memory_data_min[i]
        mem /= memory_data_max[i]
        i+=1

    i=0
    for disk in disk_data:
        disk -= disk_data_min[i]
        disk /= disk_data_max[i]
        i+=1
    # print((cpu_data[7] > 1).any())
    return cpu_data, cpu_data_max, cpu_data_min, memory_data, memory_data_max, memory_data_min, disk_data, disk_data_max, disk_data_min

