import torch
import math
import os
import wandb
import numpy as np
from torchsummary import summary


class Trainer:
    def __init__(
        self, model, criterion, optimizer, scheduler, config, train_loader, val_loader
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_loss_hist = []
        self.val_loss_hist = []

        self.curr_lr = config["LEARNING_RATE"]

        self.epoch = 0

    def init_wandb(self):
        # wandb.login(key=os.environ.)

        wandb.init(
            # set the wandb project where this run will be logged
            project=self.config["PROJECT"],
            # track hyperparameters and run metadata
            config=self.config,
        )

        # Save model architecture as a temporary file to upload to wandb
        file = open(os.path.join(wandb.run.dir, "model_arch.txt"), "w")
        file.write(str(self.model))
        file.close()

        wandb.save(os.path.join(wandb.run.dir, "model_arch.txt"))

        # If saving images for logging, update image subdirectory with name of this run
        if self.config["SAVE_IMAGES"]:
            self.config["IMAGES_SAVE_DIR"] = os.path.join(
                self.config["IMAGES_SAVE_DIR"], wandb.run.name
            )

            # Create the directory
            os.mkdir(self.config["IMAGES_SAVE_DIR"])

        # Store scheduler and optimizer type
        scheduler_string = str(self.scheduler)
        scheduler_params = self.scheduler.state_dict()
        scheduler_string_with_params = (
            f"{scheduler_string}\nParameters: {scheduler_params}"
        )

        self.config["scheduler"] = scheduler_string_with_params
        self.config["optimizer"] = str(self.optimizer)

    def train_epoch(self):
        running_loss = 0.0

        for x, y in self.train_loader:
            torch.cuda.empty_cache()

            # Remember, it's in batches.

            """

            Reset gradients to 0 so updating of weights can be done correctly.

            When we do loss.backward(), gradients are calculated. Then, optimizer.step() does gradient descent.
            For the next batch, we don't want these gradients to still be lingering (because a new input will have new gradients).
            Thus, we have to reset the gradients to 0. 

            NOTE: This is not the same as setting the weights to 0! We are just resetting the calculated gradients.
                
            """
            self.optimizer.zero_grad()

            # Calculate model outputs
            outputs = self.model(x)

            # Calculate loss
            loss = self.criterion(outputs, y)

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

        smallest_val_loss = math.inf

        # Initialize wandb
        self.init_wandb()

        val_loss = self.validate()

        print(f"Initial Val Loss: {val_loss}")

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

            print(
                f"Epoch {epoch + 1} of {EPOCHS}, Train Loss: {loss}, Val Loss: {val_loss}"
            )

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            if self.early_stop(val_loss_hist, smallest_val_loss):
                print("Stopping early - validation loss not improving.")
                break

            # Log metrics
            wandb.log(
                {
                    "epoch": epoch,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train_loss": loss,
                    "val_loss": val_loss,
                }
            )

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

            truth = None

            for x, y in data_loader:
                pred = model(x)

                loss = self.criterion(pred, y)

                # Multiplied to get aggregate loss for the batch (average done below across all batches).
                running_loss += loss.item() * pred.shape[0]

                # Store ground truth to later save
                truth = y

            # Save an image from the last pred for logging
            if (
                self.config["SAVE_IMAGES"]
                and self.epoch % self.config["IMAGES_SAVE_INTERVAL"] == 0
            ):
                pred = pred[0].cpu().numpy()

                # Save a sample validation prediction
                filename = os.path.join(self.config["IMAGES_SAVE_DIR"], str(self.epoch))
                print("SAVING", filename)
                np.save(filename, pred)

                # Save the ground truth for that prediction
                truth_filename = os.path.join(
                    self.config["IMAGES_SAVE_DIR"], str(self.epoch) + "-truth"
                )
                print("SAVING", truth_filename)
                np.save(truth_filename, truth)

            # Revert back to train mode
            model.train()

            # Calculate MSE across the batches
            return running_loss / len(data_loader.dataset)

    def adapt_lr(self, loss, new_lr, threshold):
        if loss < threshold and new_lr < self.curr_lr:
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr
            self.curr_lr = new_lr

    def early_stop(self, val_loss_hist, smallest_val_loss):
        patience = self.config["PATIENCE"]

        # Make sure there enough epochs to do early stopping.
        if len(val_loss_hist) < patience:
            return False

        # Decide if early stopping necessary
        recent_min = min(val_loss_hist[-patience:])
        if smallest_val_loss < recent_min:
            return True
