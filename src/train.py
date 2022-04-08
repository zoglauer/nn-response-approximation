import os
import torch
import torch.nn.functional as F
from src.logger import MetricMeter, TrainingLogger
from src.utils import PytorchUtils

ptu = PytorchUtils()
logger = TrainingLogger()

class Trainer:
    def __init__(self, config, model, criterion, optimizer, cone_model, log_freq=20):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cone_model = cone_model

        # hyper-parameters
        self.total_epoch = config.epoch
        ptu.set_device(config.device)

        # logging parameters
        self.log_freq = log_freq
        self.max_times_no_improvement = 20
        self.metric_monitor = config.metric_monitor
        self.model_path = config.working_dir
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        logger.set_dest_path(os.path.join(self.model_path, 'log.txt'))
        print('Logging dir:', self.model_path)

        with open(os.path.join(self.model_path, 'model_arch.txt'), 'w') as f:
            print(str(self.model), file=f)
        
        logger.print('Number of parameters in model: {}'.format(
            sum([p.numel() for p in self.model.parameters()])
        ))
        logger.print('Using {}'.format(torch.cuda.get_device_name()))
    
    
    def train(self, train_loader, val_loader):
        train_metric_meters = {'loss': MetricMeter(), 'mse': MetricMeter(), 'mae': MetricMeter()}
        val_metric_meters = {'loss': MetricMeter(), 'mse': MetricMeter(), 'mae': MetricMeter()}

        # Plot original one
        XSingle = val_loader.dataset[0]['data']
        YSingle = val_loader.dataset[0]['label']
        self.cone_model.Plot2D(XSingle, YSingle, figure_title='Original.png')
         
        epoch, times_no_improvement = 0, 0
        min_metric = float('inf')
        self.model.to(ptu.device)
        while epoch < self.total_epoch and times_no_improvement < self.max_times_no_improvement:
            self._train_epoch(train_loader, train_metric_meters)
            self._eval_epoch(val_loader, val_metric_meters)

            if epoch % self.log_freq == 0:
                logger.print('Epoch {:3d}| Train | Loss: {:.4f}   MSE: {:.4f}   MAE: {:.4f}'.format(
                    epoch, 
                    train_metric_meters['loss'].get_score(),
                    train_metric_meters['mse'].get_score(),
                    train_metric_meters['mae'].get_score()
                ))
                logger.print('Epoch {:3d}|  Val  | Loss: {:.4f}   MSE: {:.4f}   MAE: {:.4f}'.format(
                    epoch, 
                    val_metric_meters['loss'].get_score(),
                    val_metric_meters['mse'].get_score(),
                    val_metric_meters['mae'].get_score()
                ))
  
                XSingle = val_loader.dataset[0]['data']
                YSingle = self.predict(ptu.from_numpy(XSingle))
                YSingle = ptu.to_numpy(YSingle)
                self.cone_model.Plot2D(XSingle, YSingle, 
                   figure_title='Reconstructed at epoch {}'.format(epoch))
                   
                # Log the model
                val_monitor_metric = val_metric_meters[self.metric_monitor].get_score()
                if val_monitor_metric < min_metric:
                    min_metric = val_monitor_metric
                    self.save_model()
                    
                    XSingle = val_loader.dataset[0]['data']
                    YSingle = self.predict(ptu.from_numpy(XSingle))
                    YSingle = ptu.to_numpy(YSingle)
                    self.cone_model.Plot2D(XSingle, YSingle, 
                       figure_title='Best reconstruction'.format(epoch))
                        
                    times_no_improvement = 0
                else:
                    times_no_improvement += 1 
          
            epoch += 1
        
        logger.dump_logs()

        
    def _train_epoch(self, train_loader, metric_meters):
        self._reset_meters(metric_meters)
        self.model.train()
        for i, batch in enumerate(train_loader, start=1):
            data = ptu.to_device(batch['data'])
            label = ptu.to_device(batch['label'])

            self.optimizer.zero_grad()
            pred = self.model(data)            
            loss = self.criterion(pred, label)
            loss.backward()
            self.optimizer.step()

            # Logging
            pred = pred.detach()
            mse = F.mse_loss(pred, label, reduction='mean')
            mae = F.l1_loss(pred, label, reduction='mean') 
            metric_meters['loss'].update(loss.item())
            metric_meters['mse'].update(mse.item())
            metric_meters['mae'].update(mae.item())


    def _eval_epoch(self, val_loader, metric_meters=None):
        if metric_meters:
            self._reset_meters(metric_meters)

        self.model.eval()   
        for i, batch in enumerate(val_loader, start=1):  
            data = ptu.to_device(batch['data'])
            label = ptu.to_device(batch['label'])
            
            with torch.no_grad(): 
                pred = self.model(data)  
            
            loss = self.criterion(pred, label)

            # Logging
            if metric_meters:
                pred = pred.detach()
                mse = F.mse_loss(pred, label, reduction='mean')
                mae = F.l1_loss(pred, label, reduction='mean') 
                metric_meters['loss'].update(loss.item())
                metric_meters['mse'].update(mse.item())
                metric_meters['mae'].update(mae.item())


    def _reset_meters(self, meters):
        for name, meter in meters.items():
            meter.reset()


    def evaluate(self, eval_loader):
        predictions = []
        for i, batch in enumerate(eval_loader, start=1):  
            data = ptu.to_device(batch['data'])
            label = ptu.to_device(batch['label'])
            
            with torch.no_grad(): 
                pred = self.model(data)  
            
            predictions.append(pred.cpu())
            
        predictions = torch.cat(predictions, dim=0)

        return predictions

    def predict(self, data):
        with torch.no_grad():  
            data = data.unsqueeze(0)
            return self.model(data).squeeze(0)
    

    def save_model(self, save_name='best_loss.pth'):
        logger.print('Save model to {}'.format(os.path.join(self.model_path, save_name)))
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.model_path, save_name)
        )


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=ptu.device))