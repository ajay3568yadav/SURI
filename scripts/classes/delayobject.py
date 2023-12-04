class DelayObject:
    def __init__(self, width, radix, name, range_="", toggle=None,binary =False, binary_flag = 0):
        self.width = width
        self.radix = radix
        self.name = name
        self.range_ = range_
        self.binary = binary #if the object has a long binary output
        self.binary_flag = binary_flag #binary object switching value
        self.toggle = toggle if toggle is not None else {}

