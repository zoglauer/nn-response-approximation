import pickle
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset

class HealpixDataLoader():

    def __init__(self, file_name, output_dim, config, SHUFFLE):
        # Load in dataset and split into x, y categories
        x, y = self.load_data(file_name)

        # Reshape output data into (depth by width by length)
        x, y = self.reshape_data(x, y, output_dim)

        # Split into train, test, and validation arrays
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_data(x, y, config["train_pct"], config["val_pct"])

        # Create TensorDatasets
        train_set, val_set, test_set = self.create_datasets(x_train, x_val, x_test, y_train, y_val, y_test, 
                                                            config["base"], config["device"])

        # Create DataLoaders
        train_loader, val_loader, test_loader = self.create_loaders(train_set, val_set, test_set, 
                                                                    config["BATCH_SIZE"], SHUFFLE)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def load_data(self, file_name):
        f = open(file_name, "rb")
        dataset = pickle.load(f)
        f.close()

        x_combined = []
        y_combined = []

        for i in range(len(dataset)):
            x = dataset[i]['data']
            y = dataset[i]['label']

            x_combined.append(x)
            y_combined.append(y)
            
        x_combined = np.array(x_combined)
        y_combined = np.array(y_combined)

        return x_combined, y_combined
    
    def reshape_data(self, x, y, output_dim):
        depth, width, length = output_dim
        y = y.reshape((len(y), depth, width, length))

        return x, y

    def split_data(self, x, y, train_pct, val_pct):
        train_len = int(len(x) * train_pct)
        val_len = int(len(x) * val_pct)

        x_train = x[:train_len]
        x_val = x[train_len : train_len + val_len]
        x_test = x[train_len + val_len:]

        y_train = y[:train_len]
        y_val = y[train_len : train_len + val_len]
        y_test = y[train_len + val_len:]

        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def create_datasets(self, x_train, x_val, x_test, y_train, y_val, y_test, base, device):
        # Need to convert bases based on device 
        train_set = TensorDataset(torch.tensor(x_train).to(dtype=base, device=device), 
                                  torch.tensor(y_train).to(dtype=base, device=device))
        val_set = TensorDataset(torch.tensor(x_val).to(dtype=base, device=device), 
                                torch.tensor(y_val).to(dtype=base, device=device))
        test_set = TensorDataset(torch.tensor(x_test).to(dtype=base, device=device), 
                                 torch.tensor(y_test).to(dtype=base, device=device))
        
        return train_set, val_set, test_set

    def create_loaders(self, train_set, val_set, test_set, BATCH_SIZE, SHUFFLE=True):
        train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=SHUFFLE)
        val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=SHUFFLE)
        test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=SHUFFLE)

        return train_loader, val_loader, test_loader

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

