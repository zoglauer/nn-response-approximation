class MetricMeter:
    def __init__(self, name='', at=1):
        self.name = name
        self.at = at
        self.n = 0.0
        self.n_corrects = 0.0
        self.name = '{}@{}'.format(name, at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, correct, total=1):    
        self.n_corrects += correct    
        self.n += total

    def get_score(self):
        return self.n_corrects / self.n if self.n != 0 else 0.0



class TrainingLogger:
    def __init__(self, dest_path='.'):
        self.logs = []
        self.dest_path = dest_path 
    
    def set_dest_path(self, dest_path):
        self.dest_path = dest_path
    
    def reset(self, dest_path='.'):
        self.logs = [] 
        self.dest_path = dest_path 

    def print(self, epoch_msg, end='\n', show=True):
        if show:
            print(epoch_msg + ' '*20, end=end)
        self.logs.append(epoch_msg)
        self.dump_logs()
    
    def dump_logs(self):
        msg = '\n'.join(self.logs)
        with open(self.dest_path, 'w') as f:
            f.write(msg)
