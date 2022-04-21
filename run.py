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
from src.autoencoder_healpix import AutoEncoder, AutoEncoderTrainer

if __name__ == '__main__':

    # Config
    config = Config()
    config.model_type = 'fc'
    config.loss_type = 'MSELoss'
    config.metric_monitor = 'loss'
    config.lr = 1e-3
    config.dropout_rate = 0.0
    config.train_batch_size = 1024
    config.eval_batch_size = 1024
    config.epoch = 15000
    config.device = 'cuda:0'
    
    # flattened = True for using fully-connected layers, False for using conv-based layers
    config.flattened = True  
    config.filter_size = 3
    config.NSIDE = 6
    config.exp_name = 'sphere_{}_{}_NSIDE{}_datasize1024'.format(
       config.model_type, config.loss_type, config.NSIDE)
    config.working_dir = os.path.join('results', 
        '{}_{}'.format(config.exp_name, time.strftime('%m%d_%H-%M'))
    )
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
        output_dir=os.path.join(config.working_dir, 'figs'),
        NSIDE=config.NSIDE
    )
    #train_dset = cone_model.create_dataset(dataset_size=1024)
    #val_dset = cone_model.create_dataset(dataset_size=1024)   
    #f = open(f"sphere_datasets_NSIDE{config.NSIDE}.pkl", "wb"); pickle.dump((train_dset, val_dset), f); f.close()
    
    # this pkl file is generated using the above three lines of code
    f = open("sphere_datasets_NSIDE6.pkl", "rb"); train_dset, val_dset = pickle.load(f); f.close()
    
    train_loader = torch.utils.data.DataLoader(train_dset, 
        batch_size=config.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dset, 
        batch_size=config.eval_batch_size)
    

    # Model, Loss, Optimizer
    if config.model_type == 'fc':
        model_param = {
            'input_size': cone_model.InputDataSpaceSize, 
            'output_size': cone_model.OutputDataSpaceSize, 
            'dropout_rate': config.dropout_rate
        }
    else:
        model_param = {
            'input_size': cone_model.InputDataSpaceSize, 
            'num_z_grids': cone_model.gTrainingGridZ, 
            'num_xy_grids': cone_model.gTrainingGridXY
        }
    
    #Testing the autoencoder
    autoencoder = AutoEncoder(cone_model,train_loader,val_loader)
    
    autoenctrainer=AutoEncoderTrainer(train_loader,val_loader,cone_model,autoencoder,config.working_dir)
    #autoenctrainer.checkShapes()
    #autoenctrainer.tryPlot2D_toyModel()
    autoenctrainer.getDataSet()
    autoenctrainer.model_train(autoencoder)
    #autoenctrainer.test_model()

    
    model = ApproxModel(config.model_type, **model_param)
    criterion = ApproxLoss(config.loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    trainer = Trainer(config, model, criterion, optimizer, cone_model, autoenctrainer)
    trainer.train(train_loader, val_loader)
    
    