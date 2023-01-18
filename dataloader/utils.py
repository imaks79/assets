import numpy as np
import time


DEBUG = False; 
current_time = '[' + time.strftime("%H:%M:%S", time.localtime()) + ']'; 
# Код синхронизации
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]; 
# Перевод байтов
word = [1, 2 ** 8, 2 ** 16, 2 ** 24]; 


def read_data(Data):
    ## Открытие и чтение файла с двоичными данными / Чтение данных с порта напрямую (DSP)
    try: readBuffer = Data.read(Data.in_waiting);  
    except: 
        with open(Data, 'rb') as file: 
            readBuffer = file.read(); 
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8'); 
    byteCount = len(byteVec); 
    return byteVec, byteCount; 


def rewrite_data(byteVec, byteCount):
    ## Дополнение основного массива с данными
    # TODO: 2 в 15 степени не охватывает всех данных. Размер файла варьируется.
    maxBufferSize = 2 ** 19; 
    byteBuffer = np.zeros(maxBufferSize, dtype = 'uint8'); 
    byteBufferLength = 0;   
    # Перезаписываем считанные данные в byteBuffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]; 
        byteBufferLength = byteBufferLength + byteCount; 
        currentIndex = 0; 
    if DEBUG: print(f'{current_time} Размер прочитанных данных {byteCount} из {maxBufferSize} допустимых');  
    return byteBuffer, byteBufferLength, currentIndex;


def find_magic_word(byteBuffer, byteBufferLength):
    # Поиск кода синхронизации 
    if byteBufferLength > 16:
        # Индексы вхождения
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]; 
        startIdx = [];  
        # Проверка на совпадение по коду синхронизации, поиск первого вхождения, от этого индекса будет начата обработка
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+len(magicWord)]; 
            if np.all(check == magicWord): startIdx.append(loc);  
        return startIdx; 
    return 0;


def clear_data_before_magic_word(byteBuffer, byteBufferLength, startIdx):
    # Очистка массива до первого появления кода синхронизации и подсчет размера пакета
    magicOK = 0; 
    if startIdx:
    # Удаляет данные перед первым индексом
        if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
            byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]; 
            byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]), dtype = 'uint8'); 
            byteBufferLength = byteBufferLength - startIdx[0]; 
        # Проверяет на наличие ошибок
        if byteBufferLength < 0: byteBufferLength = 0; 
        # Подсчет общей длины пакета
        totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word); 
        if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0): magicOK = 1; 
        return byteBuffer, totalPacketLen, magicOK;


def clear_old_data(byteBuffer, byteBufferLength, totalPacketLen, idX):
    # Удаление уже обработанных данных
    if idX > 0 and byteBufferLength > idX:
        shiftSize = totalPacketLen; 
        # tmpBuffer = np.zeros(byteBufferLength - shiftSize, dtype = 'uint8'); 
        byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength];
        # byteBuffer = tmpBuffer; 
        byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8'); 
        byteBufferLength = byteBufferLength - shiftSize; 
        # Убедитесь, что нет ошибок с длиной буфера
        if byteBufferLength < 0:
            byteBufferLength = 0; 
        return byteBuffer, byteBufferLength; 
     

def debug_message(tlv_type, tlv_length, tlvIdx, Header):
    print(40 * '-'); 
    print(f'The {tlvIdx+1}st TLV:'); 
    print(f'\tTLV TAG: {tlv_type}'); 
    print(f'\tTLV LEN: {tlv_length}'); 
    print(40*'-'); 
    print('num detected objects:', Header['numDetectedObj']); 