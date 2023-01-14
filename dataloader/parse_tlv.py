from .analysis_tlv import *
from .utils import *

'''
Сообщения TLV
MMWDEMO_UART_MSG_DETECTED_POINTS = 1; 
MMWDEMO_UART_MSG_RANGE_PROFILE   = 2; 
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3; 
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4; 
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5; 
MMWDEMO_OUTPUT_MSG_STATS = 6; 
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO  = 7; 
MMWDEMO_OUTPUT_MSG_MAX = 8; 

Размеры
OBJ_STRUCT_SIZE_BYTES = 12; 
BYTE_VEC_ACC_MAX_SIZE = 2 ** 15; 
'''


def getHeader(byteBuffer):
    Header, idX = {}, 0; 
    Header['magicNumber'], idX      = byteBuffer[idX:idX + 8], idX + 8; 
    Header['version'], idX          = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x'), idX + 4; 
    Header['totalPacketLen'], idX   = np.matmul(byteBuffer[idX:idX + 4], word), idX + 4; 
    Header['platform'], idX         = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x'), idX + 4; 
    Header['frameNumber'], idX      = np.matmul(byteBuffer[idX:idX + 4], word), idX + 4; 
    Header['timeCpuCycles'], idX    = np.matmul(byteBuffer[idX:idX + 4], word), idX + 4; 
    Header['numDetectedObj'], idX   = np.matmul(byteBuffer[idX:idX + 4], word), idX + 4; 
    Header['numTLVs'], idX          = np.matmul(byteBuffer[idX:idX + 4], word), idX + 4; 
    Header['subFrameNumber'], idX   = np.matmul(byteBuffer[idX:idX + 4], word), idX + 4; 
    return Header, idX; 


def getMesseges(byteBuffer, idX):
    tlv_type, idX       = np.matmul(byteBuffer[int(idX):int(idX) + 4], word), idX + 4; 
    tlv_length, idX     = np.matmul(byteBuffer[int(idX):int(idX) + 4], word), idX + 4; 
    return tlv_type, tlv_length, idX; 


def get_TLV_packet(Header, byteBuffer, detObj, configParameters, idX, tlv_type):
    match tlv_type:
        case 1: detObj, dataOK, idX = get_DETECTED_POINTS(Header, byteBuffer, detObj, idX); 
        case 2: detObj, dataOK, idX = get_RANGE_PROFILE(byteBuffer, configParameters, detObj, idX); 
        case 3: detObj, dataOK, idX = get_NOISE_PROFILE(byteBuffer, configParameters, detObj, idX); 
        case 4: detObj, dataOK, idX = get_AZIMUT_STATIC_HEAT_MAP(byteBuffer, configParameters, detObj, idX); 
        case 5: detObj, dataOK, idX = get_RANGE_DOPPLER_HEAT_MAP(byteBuffer, configParameters, detObj, idX); 
    return detObj, dataOK, idX; 


def readAndParseTLVData(Data, configParameters, frames = 1):
    # Открытие и чтение файла с двоичными данными / Чтение данных с порта напрямую (DSP)
    byteVec, byteCount = read_data(Data);
    # Перезаписываем считанные данные в byteBuffer
    byteBuffer, byteBufferLength, currentIndex = rewrite_data(byteVec, byteCount); 
    while currentIndex < frames:
        # Поиск кода синхронизации 
        startIdx = find_magic_word(byteBuffer, byteBufferLength); 
        # Очистка массива до первого появления кода синхронизации и подсчет размера пакета
        byteBuffer, totalPacketLen, magicOK = clear_data_before_magic_word(byteBuffer, byteBufferLength, startIdx); 
        # Данные
        # TODO: ПРОВЕРИТЬ РАБОТОСПОСОБНОСТЬ
        detObj, frameData = np.array(), np.array(); 
        currentIndex, dataOK = 0, 0; 
        # Анализ TLV 
        if magicOK:
            # Чтение заголовка
            Header, idX = getHeader(byteBuffer); 
            # Чтение TLV сообщений
            for tlvIdx in range(Header['numTLVs']):
                tlv_type, tlv_length, idX = getMesseges(byteBuffer, idX); 
                if DEBUG: debug_message(tlv_type, tlv_length, tlvIdx, Header); 
                # Чтение данных по TLV сообщению
                detObj, dataOK, idX = get_TLV_packet(Header, byteBuffer, detObj, configParameters, idX, tlv_type); 
            # Удаление уже обработанных данных
            byteBuffer, byteBufferLength = clear_old_data(byteBuffer, totalPacketLen, idX); 
            if dataOK:
                # Store the current frame into frameData
                frameData[currentIndex] = detObj; 
                currentIndex += 1; 
    return frameData, dataOK, Header; 