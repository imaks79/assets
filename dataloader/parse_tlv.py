import numpy as np
import time 
current_time = '[' + time.strftime("%H:%M:%S", time.localtime()) + ']'; 

'''
TODO: Программа парсит только одну малую часть данных из файла, 
нужно сделать так,
чтобы, к примеру, начальный индекс не сбрасывался по окончанию, а запоминался, чтобы по новой не начинался анализ,
это только для анализа данных из файла.
'''


def readAndParseTLVData(Data, configParameters, frames = 1, DEBUG = False):
    '''
    В случае успешного парса данных возвращается:
    @return dataOK                 - Статус работы с данными = 1
    @return Header                 - Количество кадров из заголовка
    @return frameData              - Данные из файла
    
    Реализовано:
    @param MMWDEMO_UART_MSG_DETECTED_POINTS             = 1 (COMPLETED)
    @param MMWDEMO_UART_MSG_RANGE_PROFILE               = 2 (COMPLETED)
    @param MMWDEMO_OUTPUT_MSG_NOISE_PROFILE             = 3 (COMPLETED)
    @param MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP    = 5 (TEST)

    TODO: ПРОВЕРИТЬ ВЫВОД ТОЧЕК В GUI ГРАФИКОВ 2, 3, 5
    TODO: РЕАЛИЗОВАТЬ ВЫВОД ГРАФИКА 4

    Не реализовано
    @param MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP    = 4 (TODO)
    @param MMWDEMO_OUTPUT_MSG_STATS                     = 6
    @param MMWDEMO_OUTPUT_MSG_MAX                       = 8
    '''
    ## Открытие и чтение файла с двоичными данными / Чтение данных с порта напрямую (DSP)
    try:
        readBuffer = Data.read(Data.in_waiting);  
    except:
        with open(Data, 'rb') as file:
            readBuffer = file.read(); 
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8'); 
    byteCount = len(byteVec);   
    ## Дополнение основного массива с данными
    # TODO: 2 в 15 степени не охватывает всех данных. Размер файла варьируется.
    maxBufferSize = 2 ** 19; 
    if DEBUG:
        print(f'{current_time} Размер прочитанных данных {byteCount} из {maxBufferSize} допустимых'); 
    byteBuffer = np.zeros(maxBufferSize, dtype = 'uint8'); 
    byteBufferLength = 0; 
    # Перезаписываем считанные данные в byteBuffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]; 
        byteBufferLength = byteBufferLength + byteCount; 
        currentIndex = 0; 
    while currentIndex < frames:
        ## Поиск кода синхронизации 
        # Код синхронизации
        magicWord = [2, 1, 4, 3, 6, 5, 8, 7]; 
        if byteBufferLength > 16:
            # Индексы вхождения
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]; 
            startIdx = [];  
            # Проверка на совпадение по коду синхронизации, поиск первого вхождения, от этого индекса будет начата обработка
            for loc in possibleLocs:
                check = byteBuffer[loc:loc+len(magicWord)]; 
                if np.all(check == magicWord):
                    startIdx.append(loc);  
        ## Очистка массива до первого появления кода синхронизации и подсчет размера пакета
        # Перевод байтов
        word = [1, 2**8, 2**16, 2**24]; 
        magicOK = 0; 
        if startIdx:
        # Удаляет данные перед первым индексом
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]; 
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]), dtype = 'uint8'); 
                byteBufferLength = byteBufferLength - startIdx[0]; 
            # Проверяет на наличие ошибок
            if byteBufferLength < 0:
                byteBufferLength = 0; 
            # Подсчет общей длины пакета
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word); 
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1; 
        ## Анализ TLV 
        '''Сюда входит чтение шапки: header 
        чтение данных полученных радаром: всех графиков'''
        OBJ_STRUCT_SIZE_BYTES = 12; 
        BYTE_VEC_ACC_MAX_SIZE = 2**15; 
        # Сообщения TLV
        MMWDEMO_UART_MSG_DETECTED_POINTS = 1; 
        MMWDEMO_UART_MSG_RANGE_PROFILE   = 2; 
        MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3; 
        MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4; 
        MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5; 
        MMWDEMO_OUTPUT_MSG_STATS = 6; 
        MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO  = 7; 
        MMWDEMO_OUTPUT_MSG_MAX = 8; 
        # Данные
        Header = {}; 
        detObj = {}; 
        frameData = {}; 
        currentIndex = 0; 
        dataOK = 0; 
        # Чтение заголовка
        if magicOK:
            idX = 0; 
            Header['magicNumber'] = byteBuffer[idX:idX + 8]; 
            idX += 8; 
            Header['version'] = format(np.matmul(byteBuffer[idX:idX+4], word), 'x'); 
            idX += 4; 
            Header['totalPacketLen'] = np.matmul(byteBuffer[idX:idX+4], word); 
            idX += 4; 
            Header['platform'] = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x'); 
            idX += 4; 
            Header['frameNumber'] = np.matmul(byteBuffer[idX:idX + 4], word); 
            idX += 4; 
            Header['timeCpuCycles'] = np.matmul(byteBuffer[idX:idX + 4], word); 
            idX += 4; 
            Header['numDetectedObj'] = np.matmul(byteBuffer[idX:idX + 4], word); 
            idX += 4; 
            Header['numTLVs'] = np.matmul(byteBuffer[idX:idX + 4], word); 
            idX += 4; 
            Header['subFrameNumber'] = np.matmul(byteBuffer[idX:idX + 4], word); 
            idX += 4; 
            # Чтение TLV сообщений
            for tlvIdx in range(Header['numTLVs']):
                # TAG of TLV messege
                tlv_type = np.matmul(byteBuffer[int(idX):int(idX) + 4], word); 
                idX += 4; 
                # LEN of TLV messeges
                tlv_length = np.matmul(byteBuffer[int(idX):int(idX) + 4], word); 
                idX += 4; 
                if DEBUG:
                    print(40*'-'); 
                    print(f'The {tlvIdx+1}st TLV:'); 
                    print(f'\tTLV TAG: {tlv_type}'); 
                    print(f'\tTLV LEN: {tlv_length}'); 
                    print(40*'-'); 
                    print('num detected objects:', Header['numDetectedObj']); 

                # Чтение данных по TLV сообщению
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    # Инициализаци массивов
                    x = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
                    y = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
                    z = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
                    velocity = np.zeros(Header['numDetectedObj'],dtype = np.float32); 
                    compDetectedRange = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
                    detectedElevAngle = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
                    detectedAzimuth = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
                    for objectNum in range(Header['numDetectedObj']):
                        # Чтение данных для объекта
                        x[objectNum] = byteBuffer[idX:idX + 4].view(dtype = np.float32); 
                        idX += 4; 
                        y[objectNum] = byteBuffer[idX:idX + 4].view(dtype = np.float32); 
                        idX += 4; 
                        z[objectNum] = byteBuffer[idX:idX + 4].view(dtype = np.float32); 
                        idX += 4; 
                        velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype = np.float32); 
                        idX += 4; 
                        # 16 позиций по 7 целей
                        # Вычисление углов и расстояний
                        '''compDetectedRange[objectNum] = np.sqrt((x[objectNum] * x[objectNum])+(y[objectNum] * y[objectNum])+(z[objectNum] * z[objectNum])); 
                        # calculate azimuth from x, y           
                        if y[objectNum] == 0:
                            if x[objectNum] >= 0:
                                detectedAzimuth[objectNum] = 90; 
                            else:
                                detectedAzimuth[objectNum] = -90; 
                        else:
                            detectedAzimuth = np.arctan(x[objectNum] / y[objectNum]) * 180 / np.pi; 
                        # calculate elevation angle from x, y, z
                        if x[objectNum] == 0 and y[objectNum] == 0:
                            if z[objectNum] >= 0:
                                detectedElevAngle[objectNum] = 90; 
                            else: 
                                detectedElevAngle[objectNum] = -90; 
                        else:
                            detectedElevAngle[objectNum] = np.arctan(z[objectNum] / np.sqrt((x[objectNum] * x[objectNum]) + (y[objectNum] * y[objectNum]))) * 180 / np.pi
                    '''# Сохранение данных
                    detObj['x'] = x; 
                    detObj['y'] = y; 
                    detObj['z'] = z; 
                    detObj['velocity'] = velocity; 
                    # detObj['compDetectedRange'] = compDetectedRange; 
                    # detObj['detectedAzimuth'] = detectedAzimuth; 
                    # detObj['detectedElevAngle'] = detectedElevAngle; 
                    dataOK = 1; 
                
                elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                    numBytes = configParameters['numRangeBins'] * 2; 
                    payload = byteBuffer[idX:idX + numBytes]; 
                    idX += numBytes; 
                    range_profile = payload.view(dtype=np.int16); 
                    detObj['range_profile'] = range_profile; 
                    dataOK = 1; 

                elif tlv_type == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE:
                    numBytes = configParameters['numRangeBins'] * 2; 
                    payload = byteBuffer[idX:idX + numBytes]; 
                    idX += numBytes; 
                    noise_profile = payload.view(dtype=np.int16); 
                    detObj['noise_profile'] = noise_profile; 
                    dataOK = 1; 


                elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP:
                    numVirtualAntennas = 4 * 2; 
                    numBytes =  configParameters['numRangeBins'] * numVirtualAntennas; 
                    payload = byteBuffer[idX:idX + int(numBytes)]; 
                    idX += numBytes; 
                    AzimuthDoppler = payload.view(dtype = np.int16);
                    # Список в матрицу TODO
                    #  
                    detObj['azimuthDoppler'] = AzimuthDoppler; 
                    


                elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
                    numBytes = configParameters['numRangeBins'] * configParameters['numDopplerBins'] * 2; 
                    payload = byteBuffer[idX:idX + int(numBytes)]; 
                    idX += numBytes; 
                    rangeDoppler = payload.view(dtype=np.int16); 
                    # Список в матрицу
                    rangeDoppler = np.reshape(rangeDoppler, (int(configParameters['numDopplerBins']), int(configParameters['numRangeBins'])), 'F'); 
                    detObj['rangeDoppler'] = np.append(rangeDoppler[int(len(rangeDoppler)/2):], rangeDoppler[:int(len(rangeDoppler)/2)], axis = 0); 
                    # Для построения графика
                    detObj['rangeArray'] = np.array(range(configParameters['numRangeBins'])) * configParameters['rangeIdxToMeters']; 
                    detObj['dopplerArray'] = np.multiply(np.arange(-configParameters['numDopplerBins']/2, configParameters['numDopplerBins']/2), configParameters['dopplerResolutionMps']); 
                    dataOK = 1; 


                elif tlv_type == MMWDEMO_OUTPUT_MSG_STATS:
                    pass; 


                elif tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO:
                    snr = np.zeros(Header['numDetectedObj'],dtype=np.float32); 
                    noise = np.zeros(Header['numDetectedObj'],dtype=np.float32); 
                    for objectNum in range(Header['numDetectedObj']):
                        snr[objectNum] = np.matmul(byteBuffer[idX + 0:idX + 2], word[0:2]); 
                        noise[objectNum] = np.matmul(byteBuffer[idX + 2:idX + 4], word[0:2]); 
                        idX += 4; 
                    dataOK = 1; 

                
                elif tlv_type == MMWDEMO_OUTPUT_MSG_MAX:
                    pass; 

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

            if dataOK:
                # Store the current frame into frameData
                frameData[currentIndex] = detObj; 
                currentIndex += 1; 

    return frameData, dataOK, Header;