import os
import time
import torch
from src.cone_model import ToyModel3DCone
from src.loss import ApproxLoss
from src.model import ApproxModel
from src.config import Config
from src.train import Trainer
from src.utils import set_seed, ensure_dir_exists


if __name__ == '__main__':

    # Config
    config = Config()
    config.model_type = 'fc'
    config.loss_type = 'MSELoss'
    config.metric_monitor = 'loss'
    config.lr = 1e-3
    config.train_batch_size = 1024
    config.eval_batch_size = 1024
    config.epoch = 10000
    config.device = 'cuda:0'
    # config.exp_name = '{}_{}_xybins30_datasize1024'.format(config.model_type, config.loss_type)
    config.exp_name = 'kaiming_normal'
    config.working_dir = os.path.join('results', 
        '{}_{}'.format(config.exp_name, time.strftime('%m%d_%H-%M'))
    )
    ensure_dir_exists(config.working_dir)
    config.dump(os.path.join(config.working_dir, 'config.json'))
    
    set_seed(2021)

    # Dataset, DataLoader
    toy_cone = ToyModel3DCone(
        output_dir=os.path.join(config.working_dir, 'figs'),
        flattened=True
    )
    train_dset = toy_cone.create_dataset(dataset_size=1024)
    val_dset = toy_cone.create_dataset(dataset_size=1024)
    train_loader = torch.utils.data.DataLoader(train_dset, 
        batch_size=config.train_batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dset, 
        batch_size=config.eval_batch_size, num_workers=4)
    

    # Model, Loss, Optimizer
    input_size, output_size = toy_cone.InputDataSpaceSize, toy_cone.OutputDataSpaceSize
    fc_param = {'input_size': input_size, 'output_size': output_size}
    model = ApproxModel(config.model_type, **fc_param)
    criterion = ApproxLoss(config.loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    

    trainer = Trainer(config, model, criterion, optimizer, toy_cone)
    trainer.train(train_loader, val_loader)

