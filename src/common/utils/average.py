class Average:
    def __init__(self):
        self.val = 0.0
        self.count = 0

    def add(self, val):
        self.val += val
        self.count += 1

    @property
    def avg(self):
        return self.val / self.count
