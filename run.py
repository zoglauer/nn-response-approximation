import os
import time
import torch
import pickle
from src.cone_model import ToyModel3DCone, HEALPixCone
from src.loss import ApproxLoss
from src.model import ApproxModel
from src.config import Config
from src.train import Trainer
from src.utils import set_seed, ensure_dir_exists
from torchsummary import summary

if __name__ == '__main__':

    # Config
    config = Config()
    config.model_type = 'fc'
    config.loss_type = 'MSELoss'
    config.metric_monitor = 'loss'
    config.lr = 1e-2
    config.train_batch_size = 1024
    config.eval_batch_size = 1024
    config.epoch = 20000
    config.device = 'cuda:0'
    
    # True for using fully-connected layers, False for using conv-based layers
    # If flatten, the label will be of length 3600. Otherwise, it will be of shape (30, 30, 4)
    config.flattened = False  
    config.filter_size = 3
    #config.exp_name = 'test_{}_{}_prefilter{}_xybins30_datasize1024'.format(
    #    config.model_type, config.loss_type, config.filter_size)
    config.exp_name = 'results/test'.format(
       config.model_type, config.loss_type, config.filter_size)
    config.working_dir = os.path.join('results', 
        '{}_{}'.format(config.exp_name, time.strftime('%m%d_%H-%M'))
    )
    config.working_dir = config.exp_name
    ensure_dir_exists(config.working_dir)
    config.dump(os.path.join(config.working_dir, 'config.json'))
    
    set_seed(2021)

    # Dataset, DataLoader
    # cone_model = ToyModel3DCone(
    #     output_dir=os.path.join(config.working_dir, 'figs'),
    #     flattened=config.flattened,
    #     filter_size=config.filter_size
    # )
    cone_model = HEALPixCone(
        output_dir=os.path.join(config.working_dir, 'figs')
    )
    # train_dset = cone_model.create_dataset(dataset_size=1024)
    # val_dset = cone_model.create_dataset(dataset_size=1024)   
    # f = open("sphere_datasets.pkl", "wb"); pickle.dump((train_dset, val_dset), f); f.close()
    f = open("sphere_datasets.pkl", "rb"); train_dset, val_dset = pickle.load(f); f.close()
    cone_model.Plot2D(train_dset[0]['data'], train_dset[0]['label'].reshape(1, -1), 'test')
    exit()
    train_loader = torch.utils.data.DataLoader(train_dset, 
        batch_size=config.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dset, 
        batch_size=config.eval_batch_size)
    

    # Model, Loss, Optimizer
    if config.model_type == 'fc':
        model_param = {
            'input_size': cone_model.InputDataSpaceSize, 
            'output_size': cone_model.OutputDataSpaceSize
        }
    else:
        model_param = {
            'input_size': cone_model.InputDataSpaceSize, 
            'num_z_grids': cone_model.gTrainingGridZ, 
            'num_xy_grids': cone_model.gTrainingGridXY
        }
        
    model = ApproxModel(config.model_type, **model_param)

    criterion = ApproxLoss(config.loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    

    trainer = Trainer(config, model, criterion, optimizer, cone_model)
    trainer.train(train_loader, val_loader)

