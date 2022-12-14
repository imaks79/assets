from utils import *


def getHeader(byteBuffer):
    Header, idX                     = {}, 0; 
    Header['magicNumber'], idX      = get_magicNumber(byteBuffer, idX);
    Header['version'], idX          = get_version(byteBuffer, idX);
    Header['totalPacketLen'], idX   = get_totalPacketLen(byteBuffer, idX);
    Header['platform'], idX         = get_platform(byteBuffer, idX);
    Header['frameNumber'], idX      = get_frameNumber(byteBuffer, idX);
    Header['timeCpuCycles'], idX    = get_timeCpuCycles(byteBuffer, idX);
    Header['numDetectedObj'], idX   = get_numDetectedObj(byteBuffer, idX);
    Header['numTLVs'], idX          = get_numTLVs(byteBuffer, idX);
    Header['subFrameNumber'], idX   = get_subFrameNumber(byteBuffer, idX);
    return Header, idX;


def getMesseges(byteBuffer, idX):
    tlv_type, idX = get_type(byteBuffer, idX);
    tlv_length, idX = get_len(byteBuffer, idX);
    return tlv_type, tlv_length, idX;


def get_TLV_packet(Header, byteBuffer, detObj, configParameters, idX, tlv_type):
    if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS: detObj, dataOK, idX = get_DETECTED_POINTS(Header, byteBuffer, detObj, idX);
    elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE: detObj, dataOK, idX = get_RANGE_PROFILE(byteBuffer, configParameters, detObj, idX);
    elif tlv_type == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE: detObj, dataOK, idX = get_NOISE_PROFILE(byteBuffer, configParameters, detObj, idX);
    elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP: detObj, dataOK, idX = get_AZIMUT_STATIC_HEAT_MAP(byteBuffer, configParameters, detObj, idX);
    elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP: detObj, dataOK, idX = get_RANGE_DOPPLER_HEAT_MAP(byteBuffer, configParameters, detObj, idX);
    elif tlv_type == MMWDEMO_OUTPUT_MSG_STATS: pass; 
    elif tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO: pass; # get_DETECTED_POINTS_SIDE_INFO(Header, byteBuffer, idX);
    elif tlv_type == MMWDEMO_OUTPUT_MSG_MAX: pass; 
    return detObj, dataOK, idX;


def readAndParseTLVData(Data, configParameters, frames = 1):
    ## Открытие и чтение файла с двоичными данными / Чтение данных с порта напрямую (DSP)
    byteVec, byteCount = readdata(Data);
    # Перезаписываем считанные данные в byteBuffer
    byteBuffer, byteBufferLength, currentIndex = rewritedata(byteVec, byteCount);
    while currentIndex < frames:
        ## Поиск кода синхронизации 
        startIdx = find_magic_word(byteBuffer, byteBufferLength);
        ## Очистка массива до первого появления кода синхронизации и подсчет размера пакета
        byteBuffer, totalPacketLen, magicOK = clear_data_before_magic_word(byteBuffer, byteBufferLength, startIdx);
        ## Данные
        detObj, frameData = {}, {}; 
        currentIndex, dataOK = 0, 0; 
        ## Анализ TLV 
        if magicOK:
            ## Чтение заголовка
            Header, idX = getHeader(byteBuffer);
            ## Чтение TLV сообщений
            for tlvIdx in range(Header['numTLVs']):
                tlv_type, tlv_length, idX = getMesseges(byteBuffer, idX);
                if DEBUG: debug_message(tlv_type, tlv_length, tlvIdx, Header);
                ## Чтение данных по TLV сообщению
                detObj, dataOK, idX = get_TLV_packet(Header, byteBuffer, detObj, configParameters, idX, tlv_type);
            ## Удаление уже обработанных данных
            byteBuffer, byteBufferLength = clear_old_data(byteBuffer, totalPacketLen, idX)
            if dataOK:
                ## Store the current frame into frameData
                frameData[currentIndex] = detObj; 
                currentIndex += 1; 
    return frameData, dataOK, Header;

