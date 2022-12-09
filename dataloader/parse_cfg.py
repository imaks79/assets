def parseConfigFile(configFileName, numRxAnt = 4, numTxAnt = 2):
    '''
    Функци получения конфигураци из файла .cfg
    @input configFileName       -   путь к файлу конфигурации
    @return configParameters    -   параметры конфигурации
    '''
    configParameters = {};  # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]; 
    for i in config:
        
        # Split the line
        splitWords = i.split(" "); 
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2])); 
            idleTime = int(splitWords[3]); 
            rampEndTime = float(splitWords[5]); 
            freqSlopeConst = float(splitWords[8]); 
            numAdcSamples = int(splitWords[10]); 
            numAdcSamplesRoundTo2 = 1; 
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2; 
                
            digOutSampleRate = int(splitWords[11]); 
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1]); 
            chirpEndIdx = int(splitWords[2]); 
            numLoops = int(splitWords[3]); 
            numFrames = int(splitWords[4]); 
            framePeriodicity = float(splitWords[5]); 


    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops; 
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt; 
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2; 
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples); 
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"]); 
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt); 
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3); 
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt); 
    configParameters['numTxAnt'] = numTxAnt; 
    configParameters['numRxAnt'] = numRxAnt; 

    return configParameters; 
