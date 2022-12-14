import serial 
import time


def serialConfig(configFileName, DATAPORT_NAME = 'COM4', CLIPORT_NAME = 'COM3', CLI_BAUD = 115200, DATA_BAUD = 921600):
        '''Открывает порт конфигурации и порт данных'''
        CLIport = serial.Serial(CLIPORT_NAME, CLI_BAUD); 
        Dataport = serial.Serial(DATAPORT_NAME, DATA_BAUD); 
        ''' Read the configuration file and send it to the board '''
        config = [line.rstrip('\r\n') for line in open(configFileName)]; 
        for i in config:
            CLIport.write((i+'\n').encode()); 
            print(i); 
            time.sleep(0.01); 
        return CLIport, Dataport; 
