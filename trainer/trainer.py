import torch

class Trainer():
    def __init__(self, model, criterion, optimizer, config, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.config = config

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_loss_hist = []
        self.val_loss_hist = []

        self.curr_lr = config["LEARNING_RATE"]

    def train_epoch(self):
        running_loss = 0.0
        
        for x_vals, y_vals in self.train_loader:
            torch.cuda.empty_cache()

            # Remember, it's in batches. 
                
            '''

            Reset gradients to 0 so updating of weights can be done correctly.

            When we do loss.backward(), gradients are calculated. Then, optimizer.step() does gradient descent.
            For the next batch, we don't want these gradients to still be lingering (because a new input will have new gradients).
            Thus, we have to reset the gradients to 0. 

            NOTE: This is not the same as setting the weights to 0! We are just resetting the calculated gradients.
                
            '''
            self.optimizer.zero_grad()

            # Calculate model outputs
            outputs = self.model(x_vals)

            # Calculate loss
            loss = self.criterion(outputs, y_vals)

            # Calculate gradients 
            loss.backward()

            # Do gradient descent to update the weights.
            self.optimizer.step()

            # CRUCIAL: need to multiply by batch size since loss.item() will give total loss / batch_size
            running_loss += loss.item() * outputs.shape[0]
        
        return running_loss

    def train(self):
        EPOCHS = self.config["EPOCHS"]

        train_loss_hist = self.train_loss_hist
        val_loss_hist = self.val_loss_hist
        epoch = 0

        smallest_val_loss = 10000

        while epoch < EPOCHS:
            running_loss = self.train_epoch()

            # IMPORTANT: need to do train_loader.dataset to get total # training examples instead of # batches
            # len(train_loader) would just give the # of batches
            loss = running_loss / len(self.train_loader.dataset)
            train_loss_hist.append(loss)

            torch.cuda.empty_cache()

            val_loss = self.validate()
            val_loss_hist.append(val_loss)

            smallest_val_loss = min(smallest_val_loss, val_loss)

            self.adapt_lr(loss, new_lr=0.005, threshold=0.1)
            self.adapt_lr(loss, new_lr=0.0008, threshold=0.035)

            print(f'Epoch {epoch + 1} of {EPOCHS}, Train Loss: {loss}, Val Loss: {val_loss}')

            if self.early_stop(val_loss_hist, smallest_val_loss):
                print("Stopping early - validation loss not improving.")
                break
            
            epoch += 1
        
        # Store epoch in object
        self.epoch = epoch

    def validate(self):
        model = self.model
        data_loader = self.val_loader

        # Set model to evaluation mode to conserve memory 
        model.eval()

        # Don't want to waste memory on gradients
        with torch.no_grad():
            running_loss = 0.0
                
            for x, y in data_loader:
                pred = model(x)

                loss = self.criterion(pred, y)

                 # Multiplied to get aggregate loss for the batch (average done below across all batches).
                running_loss += loss.item() * pred.shape[0]
                
            # Revert back to train mode
            model.train()

            # Calculate MSE across the batches
            return running_loss / len(data_loader.dataset)
        
    def adapt_lr(self, loss, new_lr, threshold):
        if loss < threshold and new_lr < self.curr_lr:
            for g in self.optimizer.param_groups:
                g['lr'] = new_lr
        
    def early_stop(self, val_loss_hist, smallest_val_loss):
        patience = self.config["PATIENCE"]
            
        # Decide if early stopping necessary 
        recent_min = min(val_loss_hist[-patience:])
        if smallest_val_loss < recent_min:
            return True