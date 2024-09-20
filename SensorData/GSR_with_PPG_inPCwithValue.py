import serial
import datetime
import re

# 打开串口
ser = serial.Serial('COM5', 9600)

gsr_value = 0
ppg_value = 0

with open('sensor_data.txt', 'a') as output_file:
        output_file.write("Timestamp,GSR,PPG\n")

def save_data_to_file():
    global gsr_value, ppg_value
    with open('sensor_data.txt', 'a') as output_file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_file.write(f"{timestamp},{gsr_value},{ppg_value}\n")
        output_file.flush()  # 强制写入文件

def update_data():
    global gsr_value, ppg_value
    # 读取串口数据
    while ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()   # 以utf-8格式解码串口行数据
        line = line.replace('\n', '').replace('\r', '')
        if line.startswith('G'):                        # 如果以G开头：是GSR数值
            gsr_value = int(re.sub("\D", "", line))  # 这一句是把字符串中数字提取出来
        if line.startswith('S'):                      # 如果以S开头，是脉搏波数据
            ppg_value = int(re.sub("\D", "", line))
    save_data_to_file()
    
while True:
    update_data()
