import threading
import numpy as np
import serial
import binascii


class NeuroPy(object):
    __attention = 0
    __meditation = 0
    __rawValue = 0
    __delta = 0
    __theta = 0
    __lowAlpha = 0
    __highAlpha = 0
    __lowBeta = 0
    __highBeta = 0
    __lowGamma = 0
    __midGamma = 0
    __poorSignal = 0
    __blinkStrength = 0
    __heart_rate = 0
    srl = None
    __port = None
    __baudRate = None
    __history = None
    threadRun = True  # controls the running of thread
    callBacksDictionary = {}  # keep a track of all callbacks

    def __init__(self, port, person_name, task_name, task_duration, baudRate=57600):
        np.set_printoptions(suppress=True)
        self.__port, self.__baudRate = port, baudRate
        self.person_name = person_name
        self.task_name = task_name
        self.task_duration = task_duration

    def __del__(self):
        if self.srl is not None:
            self.srl.close()

    def start(self):
        """starts packetparser in a separate thread"""
        self.threadRun = True
        self.srl = serial.Serial(self.__port, self.__baudRate)
        threading.Thread(target=self.__packetParser(self.srl)).start()

    def __packetParser(self, srl):
        "packetParser runs continously in a separate thread to parse packets from mindwave and update the corresponding variables"
        # srl.open()
        while self.threadRun:

            p1 = binascii.hexlify(srl.read(1)).decode('ascii')
            p2 = binascii.hexlify(srl.read(1)).decode('ascii')
            while p1 != 'aa' or p2 != 'aa':
                p1 = p2
                p2 = binascii.hexlify(srl.read(1)).decode('ascii')
            else:
                # a valid packet is available
                payload = []
                checksum = 0
                payloadLength = int(binascii.hexlify(srl.read(1)).decode('ascii'), 16)
                for i in range(payloadLength):
                    tempPacket = binascii.hexlify(srl.read(1)).decode('ascii')
                    payload.append(tempPacket)
                    checksum += int(tempPacket, 16)
                checksum = ~checksum & 0x000000ff
                if checksum == int(binascii.hexlify(srl.read(1)).decode('ascii'), 16):
                    i = 0

                    while i < payloadLength:
                        code = payload[i]
                        if code == '02':  # poorSignal
                            i += 1
                            self.poorSignal = int(payload[i], 16)
                        elif code == '04':  # attention
                            i += 1
                            self.attention = int(payload[i], 16)
                        elif code == '05':  # meditation
                            i += 1
                            self.meditation = int(payload[i], 16)
                        elif code == '16':  # blink strength
                            i += 1
                            self.blinkStrength = int(payload[i], 16)
                        elif code == '80':  # raw value
                            i += 1  # for length/it is not used since length =1 byte long and always=2
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            self.rawValue = val0 * 256 + int(payload[i], 16)
                            if self.rawValue > 32768:
                                self.rawValue -= 65536
                        elif code == '03':
                            i += 1
                            self.__heart_rate = int(payload[i], 16)
                        elif code == '83':  # ASIC_EEG_POWER
                            i += 1  # for length/it is not used since length =1 byte long and always=2
                            # delta:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.delta = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # theta:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.theta = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # lowAlpha:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.lowAlpha = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # highAlpha:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.highAlpha = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # lowBeta:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.lowBeta = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # highBeta:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.highBeta = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # lowGamma:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.lowGamma = val0 * 65536 + val1 * 256 + int(payload[i], 16)
                            # midGamma:
                            i += 1
                            val0 = int(payload[i], 16)
                            i += 1
                            val1 = int(payload[i], 16)
                            i += 1
                            self.midGamma = val0 * 65536 + val1 * 256 + int(payload[i], 16)

                            self.updateHistory()
                        i += 1

    def stop(self):

        self.threadRun = False
        if self.srl is not None:
            self.srl.close()

    def setCallBack(self, variable_name, callback_function):
        self.callBacksDictionary[variable_name] = callback_function

    # setting getters and setters for all variables

    # attention
    @property
    def attention(self):
        "Get value for attention"
        return self.__attention

    @attention.setter
    def attention(self, value):
        self.__attention = value
        if "attention" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["attention"](self.__attention)

    # meditation
    @property
    def meditation(self):
        "Get value for meditation"
        return self.__meditation

    @meditation.setter
    def meditation(self, value):
        self.__meditation = value
        if "meditation" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["meditation"](self.__meditation)

    # rawValue
    @property
    def rawValue(self):
        "Get value for rawValue"
        return self.__rawValue

    @rawValue.setter
    def rawValue(self, value):
        self.__rawValue = value
        if "rawValue" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["rawValue"](self.__rawValue)

    # delta
    @property
    def delta(self):
        "Get value for delta"
        return self.__delta

    @delta.setter
    def delta(self, value):
        self.__delta = value
        if "delta" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["delta"](self.__delta)

    # theta
    @property
    def theta(self):
        "Get value for theta"
        return self.__theta

    @theta.setter
    def theta(self, value):
        self.__theta = value
        if "theta" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["theta"](self.__theta)

    # lowAlpha
    @property
    def lowAlpha(self):
        "Get value for lowAlpha"
        return self.__lowAlpha

    @lowAlpha.setter
    def lowAlpha(self, value):
        self.__lowAlpha = value
        if "lowAlpha" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["lowAlpha"](self.__lowAlpha)

    # highAlpha
    @property
    def highAlpha(self):
        "Get value for highAlpha"
        return self.__highAlpha

    @highAlpha.setter
    def highAlpha(self, value):
        self.__highAlpha = value
        if "highAlpha" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["highAlpha"](self.__highAlpha)

    # lowBeta
    @property
    def lowBeta(self):
        "Get value for lowBeta"
        return self.__lowBeta

    @lowBeta.setter
    def lowBeta(self, value):
        self.__lowBeta = value
        if "lowBeta" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["lowBeta"](self.__lowBeta)

    # highBeta
    @property
    def highBeta(self):
        "Get value for highBeta"
        return self.__highBeta

    @highBeta.setter
    def highBeta(self, value):
        self.__highBeta = value
        if "highBeta" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["highBeta"](self.__highBeta)

    # lowGamma
    @property
    def lowGamma(self):
        "Get value for lowGamma"
        return self.__lowGamma

    @lowGamma.setter
    def lowGamma(self, value):
        self.__lowGamma = value
        if "lowGamma" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["lowGamma"](self.__lowGamma)

    # midGamma
    @property
    def midGamma(self):
        "Get value for midGamma"
        return self.__midGamma

    @midGamma.setter
    def midGamma(self, value):
        self.__midGamma = value
        if "midGamma" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["midGamma"](self.__midGamma)

    # poorSignal
    @property
    def poorSignal(self):
        "Get value for poorSignal"
        return self.__poorSignal

    @poorSignal.setter
    def poorSignal(self, value):
        self.__poorSignal = value
        if "poorSignal" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["poorSignal"](self.__poorSignal)

    # blinkStrength
    @property
    def blinkStrength(self):
        "Get value for blinkStrength"
        return self.__blinkStrength

    @blinkStrength.setter
    def blinkStrength(self, value):
        self.__blinkStrength = value
        if "blinkStrength" in self.callBacksDictionary:  # if callback has been set, execute the function
            self.callBacksDictionary["blinkStrength"](self.__blinkStrength)




    '''Appends the most recent read values to a local array'''

    def updateHistory(self):
        if self.__history is None:  # create it
            self.__history = np.array([[self.delta, self.theta, self.lowAlpha, self.highAlpha, self.lowBeta,
                                        self.highBeta, self.lowGamma, self.midGamma,self.attention,self.meditation,
                                        self.rawValue,self.blinkStrength]])
        else:
            self.__history=np.append(self.__history, [[self.delta, self.theta, self.lowAlpha, self.highAlpha, self.lowBeta,
                                        self.highBeta, self.lowGamma, self.midGamma,self.attention,self.meditation,
                                                       self.rawValue,self.blinkStrength]], axis=0)

    '''Saves all read values to csv'''
    def save(self):
        print('Saving data...')
        np.savetxt('records/'+self.person_name + '_' + self.task_name + '_' + self.task_duration + ".csv", self.__history,
                   delimiter=',',fmt='%.3f')
        print('Saved')
