class AverageMeter():
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val):
        self.val = val
        self.avg = val
        self.sum = val
        self.count = 1
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def avg(self):
        return self.avg

    def sum(self):
        return self.sum
