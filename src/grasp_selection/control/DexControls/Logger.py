from datetime import datetime
from DexConstants import DexConstants
class Logger:

    DEFAULT_NAME = "Zeke.log"

    @staticmethod
    def start(name = DEFAULT_NAME):
        if DexConstants.LOGGING:
            with open(name, 'w') as f:
                f.write('Logger time started at: ' + str(datetime.now()) + '\n')

    @staticmethod
    def log(label, data, name):
        if DexConstants.LOGGING:
            if data is None and label is None:
                return
            text = str(datetime.now()) + " | " + str(label)
            if data is not None:
                text += " : " + str(data)
            text += '\n'
            with open(name, 'a') as f:
                f.write(text)