import numpy as np
import os


class Parser():
    def __init__(self, root_directory):
        self.root_directory = root_directory;
        # Список доступных для чтения файлов
        self.files = list(filter(lambda x: x.endswith('.bin'), os.listdir(root_directory)));
        
    def raw_to_adc(self, file_idx:int = 0, numRX:int = 4, numADCSamples:int = 256, numADCBits:int = 16, MIMO:bool = False):    
        name_of_file = self.files[file_idx];
        path2file = os.path.join(self.root_directory, name_of_file);
        with open(path2file, 'rb') as file:
            read_data = np.array(np.frombuffer(file.read(), dtype = np.int16));
        isReal = 0;
        fileSize = read_data.shape[0];
        # if 12 or 14 bits ADC per sample compensate for sign extension
        if numADCBits != 16:
            l_max = 2**(numADCBits - 1) - 1;
            read_data[read_data > l_max] -= 2 ** numADCBits;
        # real data reshape, filesize = numADCSamples * numChirps
        if isReal:
            numChirps = int(fileSize / numADCSamples / numRX);
            #create column for each chirp
            LVDS = np.reshape(read_data, (numADCSamples * numRX, numChirps), order='F').transpose();
        else:
            # for complex data
            # filesize = 2 * numADCSamples * numChirps
            numChirps = int(fileSize / 2 / numADCSamples / numRX);
            LVDS = np.zeros(int(fileSize / 2)).astype(np.complex);
            # combine real and imaginary part into complex data
            # read in file: 2I is followed by 2Q
            LVDS[::2] = read_data[::4] + np.complex(0, 1) * read_data[2::4];
            LVDS[1::2] = read_data[1::4] + np.complex(0, 1) * read_data[3::4];
            # create column for each chirp
            # each row is data from one chirp
            LVDS = np.reshape(LVDS, (numADCSamples * numRX, numChirps), order = 'F').transpose();
        # organize data per RX
        adcData = np.zeros((numRX, numChirps * numADCSamples)).astype(np.complex);
        for row in range(numRX):
            for i in range(numChirps):
                adcData[row, i * numADCSamples:((i + 1) * numADCSamples)] = LVDS[i, row * numADCSamples:((row + 1) * numADCSamples)];

        numFrames = numChirps // numADCSamples * 2;
        numChirps = fileSize // (numChirps * numADCSamples) * numADCBits;
        
        self.numChannels = 1 * numRX;
        self.numFrames = numFrames;
        self.numChirps = numChirps;
        self.numADCSamples = numADCSamples;
        self.adcData = adcData;
    
    def getFrames(self, channel):
        '''Возвращает данные вида: data[numFrames * numChirps * numADCSamples][numFrames]'''
        return channel[:].reshape((-1, self.numFrames), order = 'F');
    
    def getChirps(self, frames):
        '''Возвращает данные вида: chirps[numADCSaples][numChirps * numFrames]'''
        tmp = [];
        for i in range(self.numFrames):
            tmp.append(frames[:, i].reshape((self.numADCSamples, self.numChirps), order = 'F'));
        return np.hstack(tmp);
    
    def organize(self, file_idx:int = 0):
        self.raw_to_adc(file_idx = file_idx);
        data = self.adcData;
        '''Возвращает данные вида: data[numADCSamples][numChannels][numChirps * numFrames]'''
        tmp_1, tmp_2 = list(), list();
        for numChannel in range(self.numChannels):
            tmp_1.append(self.getFrames(data[numChannel]));
        for frames in tmp_1:
            tmp_2.append(self.getChirps(frames));

        # TODO: Надо бы переделать
        self.out_data =  np.stack((
                                    tmp_2[0], 
                                    tmp_2[1], 
                                    tmp_2[2], 
                                    tmp_2[3]), axis = 1).transpose();
        self.out_data = self.out_data.reshape(self.numFrames, self.numChirps, self.numChannels, self.numADCSamples, order = 'C');
        
        return self.out_data;