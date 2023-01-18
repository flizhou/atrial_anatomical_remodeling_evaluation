# constants

class constants(object):
    def __init__(self):
        super().__init__()
        self.DIMENSION = 3
        # the number of spatial transformation
        self.N = 3
        self.RESOLUTION = 0.2
        self.CT_RAW = r'data/raw/ct'
        self.CT_PROCESSED = r'data/processed/ct'
