import json

class Config:
    def __init__(self):
        pass

    def to_dict(self):
        return vars(self)
    
    def from_dict(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)
    
    def from_argparse(self, args):
        self.from_dict(vars(args))

    def dump(self, path):
        config = vars(self)
        with open(path, 'w') as f:
            json.dump(config, f, indent='    ')
    
    def load(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        
        for k, v in config.items():
            setattr(self, k, v)

    def __str__(self):
        config = vars(self)
        return '\n'.join([f'  - {k}: {v}' for k, v in config.items()])



if __name__ == '__main__':

    config = Config()
    print(config)



