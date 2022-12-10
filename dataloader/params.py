import time

DEBUG = False;

current_time = '[' + time.strftime("%H:%M:%S", time.localtime()) + ']'; 

# Код синхронизации
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]; 

# Перевод байтов
word = [1, 2 ** 8, 2 ** 16, 2 ** 24]; 

# Размеры
OBJ_STRUCT_SIZE_BYTES = 12; 
BYTE_VEC_ACC_MAX_SIZE = 2 ** 15; 

# Сообщения TLV
MMWDEMO_UART_MSG_DETECTED_POINTS = 1; 
MMWDEMO_UART_MSG_RANGE_PROFILE   = 2; 
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3; 
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4; 
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5; 
MMWDEMO_OUTPUT_MSG_STATS = 6; 
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO  = 7; 
MMWDEMO_OUTPUT_MSG_MAX = 8; 