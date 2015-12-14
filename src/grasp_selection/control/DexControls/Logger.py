from datetime import datetime
from DexConstants import DexConstants

class Logger:

    @staticmethod
    def _ensure_suffix(name):
        if not name.endswith(".log"):
            name += ".log"
        return name

    @staticmethod
    def clear(name):
        name = Logger._ensure_suffix(name)
        if DexConstants.LOGGING:
            with open(name, 'w') as f:
                f.write('Logger time started at: ' + str(datetime.now()) + '\n')

    @staticmethod
    def log(label, data, name):
        name = Logger._ensure_suffix(name)
        if DexConstants.LOGGING:
            if data is None and label is None:
                return
            text = str(datetime.now()) + " | " + str(label)
            if data is not None:
                text += " : " + str(data)
            text += '\n'
            with open(name, 'a') as f:
                f.write(text)