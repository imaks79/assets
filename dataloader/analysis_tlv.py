import numpy as np


# Перевод байтов
word = [1, 2 ** 8, 2 ** 16, 2 ** 24]; 


def get_DETECTED_POINTS(Header, byteBuffer, detObj, idX):
    # Инициализаци массивов
    x = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
    y = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
    z = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
    velocity = np.zeros(Header['numDetectedObj'],dtype = np.float32); 
    compDetectedRange = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
    detectedElevAngle = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
    detectedAzimuth = np.zeros(Header['numDetectedObj'], dtype = np.float32); 
    for objectNum in range(Header['numDetectedObj']):
        # Чтение данных объекта
        x[objectNum], idX = byteBuffer[idX:idX + 4].view(dtype = np.float32), idX + 4; 
        y[objectNum], idX = byteBuffer[idX:idX + 4].view(dtype = np.float32), idX + 4; 
        z[objectNum], idX = byteBuffer[idX:idX + 4].view(dtype = np.float32), idX + 4; 
        velocity[objectNum], idX = byteBuffer[idX:idX + 4].view(dtype = np.float32), idX + 4; 
        # 16 позиций по 7 целей
        # Вычисление углов и расстояний
        # compDetectedRange[objectNum] = np.sqrt((x[objectNum] * x[objectNum])+(y[objectNum] * y[objectNum])+(z[objectNum] * z[objectNum])); 
        # # calculate azimuth from x, y           
        # if y[objectNum] == 0:
        #     if x[objectNum] >= 0:
        #         detectedAzimuth[objectNum] = 90; 
        #     else:
        #         detectedAzimuth[objectNum] = -90; 
        # else:
        #     detectedAzimuth = np.arctan(x[objectNum] / y[objectNum]) * 180 / np.pi; 
        # # calculate elevation angle from x, y, z
        # if x[objectNum] == 0 and y[objectNum] == 0:
        #     if z[objectNum] >= 0:
        #         detectedElevAngle[objectNum] = 90; 
        #     else: 
        #         detectedElevAngle[objectNum] = -90; 
        # else:
        #     detectedElevAngle[objectNum] = np.arctan(z[objectNum] / np.sqrt((x[objectNum] * x[objectNum]) + (y[objectNum] * y[objectNum]))) * 180 / np.pi
    # Сохранение данных
    detObj['x'], detObj['y'], detObj['z'], detObj['velocity'], dataOK = x, y, z, velocity, 1; 
    # detObj['compDetectedRange'] = compDetectedRange; 
    # detObj['detectedAzimuth'] = detectedAzimuth; 
    # detObj['detectedElevAngle'] = detectedElevAngle; 
   
    return detObj, dataOK, idX;


def get_RANGE_PROFILE(byteBuffer, configParameters, detObj, idX):
    numBytes = configParameters['numRangeBins'] * 2; 
    payload = byteBuffer[idX:idX + numBytes]; 
    idX += numBytes; 
    range_profile = payload.view(dtype=np.int16); 
    detObj['range_profile'] = range_profile; 
    dataOK = 1; 
    return detObj, dataOK, idX;


def get_NOISE_PROFILE(byteBuffer, configParameters, detObj, idX):
    numBytes = configParameters['numRangeBins'] * 2; 
    payload = byteBuffer[idX:idX + numBytes]; 
    idX += numBytes; 
    noise_profile = payload.view(dtype=np.int16); 
    detObj['noise_profile'] = noise_profile; 
    dataOK = 1; 
    return detObj, dataOK, idX;


def get_AZIMUT_STATIC_HEAT_MAP(byteBuffer, configParameters, detObj, idX):
    numVirtualAntennas = 4 * 2; 
    numBytes =  configParameters['numRangeBins'] * numVirtualAntennas; 
    payload = byteBuffer[idX:idX + int(numBytes)]; 
    idX += numBytes; 
    AzimuthDoppler = payload.view(dtype = np.int16);
    # Список в матрицу TODO
    detObj['azimuthDoppler'] = AzimuthDoppler; 
    dataOK = 1;
    return detObj, dataOK, idX;


def get_RANGE_DOPPLER_HEAT_MAP(byteBuffer, configParameters, detObj, idX):
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
    return detObj, dataOK, idX;


def get_STATS():
    pass;

    
def get_DETECTED_POINTS_SIDE_INFO(Header, byteBuffer, idX):
    snr = np.zeros(Header['numDetectedObj'],dtype=np.float32); 
    noise = np.zeros(Header['numDetectedObj'],dtype=np.float32); 
    for objectNum in range(Header['numDetectedObj']):
        snr[objectNum] = np.matmul(byteBuffer[idX + 0:idX + 2], word[0:2]); 
        noise[objectNum] = np.matmul(byteBuffer[idX + 2:idX + 4], word[0:2]); 
        idX += 4; 
    dataOK = 1; 


def get_MAX():
    pass;