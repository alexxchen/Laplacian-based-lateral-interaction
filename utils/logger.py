import pandas as pd
from collections import defaultdict

class logger():
    def __init__(self):
        self.summary = defaultdict(list)

    def add(self, names, values):
        for name, value in zip(names, values):
            self.summary[name].append(value)


    def print(self):
        print(self.summary)


    def write(self, path, round):
        pd.DataFrame(self.summary).to_csv(path + '/' + 'log_%d.csv'%round)


    def reset(self):
        self.summary = defaultdict(list)