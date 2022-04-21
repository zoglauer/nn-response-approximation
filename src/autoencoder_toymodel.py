from math import inf
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import os
from skimage.util import random_noise
from src.utils import PytorchUtils

ptu = PytorchUtils()

class AutoEncoder(torch.nn.Module):
  
  '''
  class AutoEncoder initializes the encoder and decoder layers for the conv net based denoising autoencoder.
  Between linear FC layer based autoencoder and conv net based autoencoder, the conv net based performed better.
  
  Class variables:
      - self.toymodel - Holds reference to the toymodel generated using the ToyModel3DCone class from src/cone_model.py
      - self.encoder - Defines the Sequential layer combinations of the encoder section of the autoencoder
      - self.decoder - Defines the Sequential layer combinations of the decoder section of the autoencoder

  Class functions:
      - self.forward() - Defines the forward pass of the autoencoder neural network.
      
  '''
  def __init__(self,toy_model):
    super().__init__()

    self.toymodel=toy_model

    #Conv2D based autoencoder
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(1,16,7,stride=1,padding=1),
      torch.nn.Dropout(0.1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(16,32,5,stride=1,padding=1),
      torch.nn.Dropout(0.1),
      torch.nn.ReLU(),
    )

    self.decoder = torch.nn.Sequential(
      
      torch.nn.ConvTranspose2d(32,16,5,stride=1,padding=1),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d(16,1,7,stride=1,padding=1),
      torch.nn.ReLU(),
    )
    

  def forward(self,x):
    #print()
    for layer in self.encoder:
      x=layer(x)
      #print(layer)
      #print("Shape now: ", x.shape)
      #print()

    for layer in self.decoder:
      x=layer(x)
      #print(layer)
      #print("Shape now: ", x.shape)
      #print()

    return x

class AutoEncoderTrainer():
  def __init__(self,train_loader,val_loader,toy_model,autoencmodel,log_dir,log_file="autoenc_logs.txt"):
    
    '''
    class AutoEncoderTrainer contains class variables and functions to train the autoencoder model

    Class variables:
        - self.toymodel - Holds reference to the toymodel generated using the ToyModel3DCone class from src/cone_model.py
        - self.train_dataset - Holds reference to the training data generated using the self.toymodel object
        - self.val_dataset - Holds reference to the validation data generated using the self.toymodel object
        - self.device - Set device to 'cuda:0' for faster training if available, else default to 'cpu'
        - self.model - Holds reference to the autoencoder created using the AutoEncoder class (previously defined in this file)
        - self.criterion - Defines the loss function for training, validation and testing; Currently using MSE loss
        - self.optimizer - Defines the optimizer used for training
              - learning rate (lr) - 0.0001
              - weight decay rate (weight_decay) - 1e-6
        
    Class functions:
        - self.model_summary() - Nested within __init__; Stores autoencoer model summary in .txt file
        - self.getDataset() - Generate train and validation datasets
        - self.checkShapes() - Explore shapes of the self.train_dataset and self.val_dataset and their contents
        - self.plot2D_oneslice() - Plot the contour plot of X and Y meshgrid for Z values along one slice of Z
        - self.plot2D_fourslice() - Plot the contour plot of X and Y meshgrid for Z values across 4 slices of Z as multiple subplots within single frame (Original image response format)
        - self.add_noise() - Adding noise to noise-free data to generate training and validation datasets for the autoencoder
        - self.model_train() - Function to train and validate the model
        - self.model_epoch() - Function to define operations for 1 epoch of model training
        - self.val_epoch() - Function to define operations for 1 epoch of validation
        - self.test_model() - Function to test performance of autoencoder and obtain prediction output from trained autoencoder
        - self.autoenc_logs() - Function to dump logs of autoencoder trainer
    '''

    self.train_dataset=train_loader.dataset
    self.val_dataset=val_loader.dataset
    self.toymodel=toy_model
    self.log_dir=log_dir
    self.log_file=os.path.join(self.log_dir, log_file)
    #self.model_file=os.path.join(self.log_dir,"autoenc_model.pth")
    self.model_file="./autoenc_model.pth"

    if torch.cuda.is_available()==True:
        self.device="cuda:0"
    else:
        self.device ="cpu"
    
    self.model=autoencmodel.to(self.device)
    self.criterion=torch.nn.MSELoss()
    self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.0001,weight_decay=1e-6)

    def model_summary():
      with open(os.path.join(self.log_dir,"autoenc_summary.txt"),"w") as file1:
        file1.write(str(self.model))
    
    model_summary()

  def getDataSet(self):

    self.x_train_data = np.zeros((len(self.train_dataset)*8,self.train_dataset[0]['label'].shape[0],self.train_dataset[0]['label'].shape[1]))
    self.x_val_data = np.zeros((len(self.val_dataset)*8,self.val_dataset[0]['label'].shape[0],self.val_dataset[0]['label'].shape[1]))
    self.y_train_data = np.zeros((len(self.train_dataset)*8,self.train_dataset[0]['label'].shape[0],self.train_dataset[0]['label'].shape[1]))
    self.y_val_data = np.zeros((len(self.val_dataset)*8,self.val_dataset[0]['label'].shape[0],self.val_dataset[0]['label'].shape[1]))
    pos=0
    for i in range(len(self.train_dataset)*4):

      train_img=self.train_dataset[i//4]['label'][:,:,i%4]
      val_img=self.val_dataset[i//4]['label'][:,:,i%4]
      
      noise=['gaussian','poisson']
      for j in noise:
        (self.y_train_data[pos],self.x_train_data[pos])=self.add_noise(train_img,j)
        (self.y_val_data[pos],self.x_val_data[pos])=self.add_noise(val_img,j)
        pos+=1
    
    self.x_train_data=torch.tensor(self.x_train_data)
    self.y_train_data=torch.tensor(self.y_train_data)
    self.x_val_data=torch.tensor(self.x_val_data)
    self.y_val_data=torch.tensor(self.y_val_data)

    self.train_data = TensorDataset(self.x_train_data,self.y_train_data)
    self.val_data = TensorDataset(self.x_val_data,self.y_val_data)

    self.trainloader=DataLoader(self.train_data,batch_size=64,shuffle=True)
    self.valloader=DataLoader(self.val_data,batch_size=64,shuffle=True)


  def checkShapes(self):
    print(len(self.train_dataset))
    print(len(self.val_dataset))
    print(len(self.train_dataset[0]['data']))
    print(len(self.val_dataset[0]['data']))
    print((self.train_dataset[0]['label']).shape)
    print((self.val_dataset[0]['label']).shape)
    print()

  
  def plot2D_oneslice(self,Z=None,title="autoencodder_plot_test"):
    
    XV, YV = np.meshgrid(self.toymodel.gGridCentersXY, self.toymodel.gGridCentersXY)
    if Z is None:
      Z = np.zeros(shape=(self.toymodel.gTrainingGridXY, self.toymodel.gTrainingGridXY))
      Z=self.y_train_data[4]

    fig = plt.figure(0)
    plt.contourf(XV, YV, Z)
    plt.savefig(title)
    plt.close()
    #plt.show()


  def plot2D_fourslice(self,Z=None,title="autoencoder_fourslice_plot_test",output_dir="."):
    XV, YV = np.meshgrid(self.toymodel.gGridCentersXY, self.toymodel.gGridCentersXY)
    if Z is None:
      print ("You need to pass some 3D Z of the form (4, height, width)!!!!")
      print()
      return Z
    
    fig = plt.figure(0)
    plt.clf()
    plt.subplots_adjust(hspace=0.5)

    for i in range(1, 5):    
      zGridElement = int((i-1)*self.toymodel.gTrainingGridZ/4)
      ZFinal=Z[i-1]
      ax = fig.add_subplot(2, 2, i)
      ax.set_title('Slice through z={}'.format(self.toymodel.gGridCentersZ[zGridElement]))
      contour = ax.contourf(XV, YV, ZFinal)

    plt.ion()
    # plt.show()
    # plt.pause(0.001)
        
    plt.savefig(os.path.join(output_dir,title))
    plt.close()
    #return Z

  
  def add_noise(self,img,noise_type="gaussian"):
  
    origImg=img.copy()
  
    if noise_type=="gaussian":
      noise=np.random.normal(0,0.0000001,img.shape)
      img=img+noise
      return (origImg,img)

    elif noise_type=="poisson":
      img=random_noise(origImg,mode='poisson')
      return(origImg,img)

    elif noise_type=="speckle":
      noise=np.random.randn(img.shape)
      img=img+img*noise
      return (origImg,img)   
  
  def model_train(self,model):
    epochs=300
    l=len(self.trainloader)
    losslist=list()
    train_epochloss=0
    val_epochloss=0
    train_running_loss=0
    val_running_loss=0
    for epoch in range(epochs):
      #print()
      #print("Entering Epoch: ",epoch)
      #print()
      train_running_loss=self.train_epoch()
      val_running_loss=self.val_epoch()

      #-----------------Log-------------------------------
      train_epochloss = train_running_loss/l
      val_epochloss = val_running_loss/len(self.valloader)
      train_str = "epoch: {}/{}, Train MSE Loss:{}".format(epoch,epochs,train_epochloss)
      val_str = "epoch: {}/{}, Val MSE Loss:{}".format(epoch,epochs,val_epochloss)
      print(train_str)
      print(val_str)
      print()

      if epoch == 0:
        losslist.append(train_epochloss)
        print("#################################################")
        print("Saving model with least loss till now in {} =====>".format(self.model_file))
        print("#################################################")
        torch.save(self.model.state_dict(), self.model_file)
        self.autoenc_logs(train_str,val_str,"w")

      else:
        if train_epochloss < losslist[-1]:
          #print("Train Loss in epoch {}: {}".format(epoch,train_epochloss))
          print("Least loss till now: ", losslist[-1])
          losslist.append(train_epochloss)
          print("#################################################")
          print("Saving model with least loss till now in {} =====>".format(self.model_file))
          print("#################################################")
          torch.save(self.model.state_dict(), self.model_file)
          self.autoenc_logs(train_str,val_str,"a")

      print()

  def train_epoch(self):
    running_loss=0.0

    for input_img,true_img in tqdm((self.trainloader)):
        #print(input_img.shape)
        #print()
        #print(true_img.shape)
        #print()
        
        #If using Conv2D based autoencoder
        input_img=input_img.view(input_img.shape[0], 1, input_img.shape[1] , input_img.shape[2])
        true_img=true_img.view(true_img.shape[0], 1, true_img.shape[1], true_img.shape[2])
        input_img=input_img.type(torch.FloatTensor).to(self.device)
        true_img=true_img.type(torch.FloatTensor).to(self.device)
        #print(input_img.shape)
        #print()
        #print(clean.shape)
        #print()
    
        #-----------------Forward Pass----------------------
        output=self.model(input_img)
        #print(output.shape)
        #print()
        loss=self.criterion(output,true_img)

        #-----------------Backward Pass---------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        running_loss+=loss.item()
    return running_loss
  

  def val_epoch(self):
    running_loss=0.0
    self.model.eval()

    for input_img,true_img in tqdm((self.valloader)):
        #print("Before reshaping input and output")
        #print("Output image shape: ",true_img.shape)
        #print("Input image shape: ",input_img.shape)
        #print()
        
        #If using Conv2D based autoencoder
        input_img=input_img.view(input_img.shape[0], 1, input_img.shape[1] , input_img.shape[2])
        true_img=true_img.view(true_img.shape[0], 1, true_img.shape[1], true_img.shape[2])
        input_img=input_img.type(torch.FloatTensor).to(self.device)
        true_img=true_img.type(torch.FloatTensor).to(self.device)
        #print("After reshaping input and output")
        #print("Output image shape: ",true_img.shape)
        #print("Input image shape: ",input_img.shape)
        #print()
    
        with torch.no_grad():
          output=self.model(input_img)
        #print("Output shape: ", output.shape)
        #print()
        loss=self.criterion(output,true_img)

        running_loss+=loss.item()

    self.model.train()
    return running_loss


  def test_model(self,input_imgs=None,epoch=-1):
    
    currModel=self.model
    currModel.load_state_dict(torch.load(self.model_file,map_location=torch.device(self.device)))
    currModel.eval()
    
    if input_imgs is None: 
      #This is if we are simply testing out how the model performs on random validation dataset members
      #Hence, both input and true output to the autoencoder are considered

      test_imgs=[]
      val_data=self.valloader.dataset
      test_imgs_idx=np.random.randint(0,len(val_data),size=4)
      for i in range(len(test_imgs_idx)):
        input_img,true_img=self.x_val_data[test_imgs_idx[i]],self.y_val_data[test_imgs_idx[i]]
        test_imgs.append((input_img,true_img))

      with torch.no_grad():
        for idx in range(len(test_imgs)):
          (input_img,true_img)=test_imgs[idx]
          #print("Before reshaping input and output")
          #print("Output image shape: ",true_img.shape)
          #print("Input image shape: ",input_img.shape)
          #print()
  
          #If using Conv2D based autoencoder
          input_img=input_img.view(1, 1, input_img.shape[0] , input_img.shape[1])
          true_img=true_img.view(1, 1, true_img.shape[0], true_img.shape[1])
          input_img=input_img.type(torch.FloatTensor).to(self.device)
          true_img=true_img.type(torch.FloatTensor).to(self.device)

          #print("After reshaping input and output")
          #print("Output image shape: ",true_img.shape)
          #print("Input image shape: ",input_img.shape)
          #print()

          output=self.model(input_img)
          loss=self.criterion(output,true_img)

          output=output.view(1,30,30)
          output=output.permute(1,2,0).squeeze(2)
          output=output.detach().cpu().numpy()
          #print(output.shape)
          self.plot2D_oneslice(Z=output,title="test_autoenc_output"+str(idx))

          input_img=input_img.view(1,30,30)
          input_img=input_img.permute(1,2,0).squeeze(2)
          input_img=input_img.detach().cpu().numpy()
          self.plot2D_oneslice(Z=input_img,title="test_autoenc_input"+str(idx))

          true_img=true_img.view(1,30,30)
          true_img=true_img.permute(1,2,0).squeeze(2)
          true_img=true_img.detach().cpu().numpy()
          self.plot2D_oneslice(Z=true_img,title="test_autoenc_original"+str(idx))

          print()
          print("Loss for current test image(s) at index {} of test set: {} ".format(idx,loss.item()))
          #print("SNR for given test image: ", 10*np.log(np.mean(output)/np.sqrt(loss.item()))) #Not sure how good a metric is SNR
          currModel.train()
    else:        
      
      with torch.no_grad():
        #print(input_imgs.shape)
        ZFinal = np.zeros(shape=((4,self.toymodel.gTrainingGridXY, self.toymodel.gTrainingGridXY)))
        for idx in range(len(input_imgs)):
          #print("Handling slice {} of 4".format(idx))
          input_img=input_imgs[:][:][idx]
          #print("Before reshaping input and output")
          #print("Input image shape: ",input_img.shape)
          #print()

          #If using Conv2D based autoencoder
          input_img=input_img.view(1, 1, input_img.shape[0] , input_img.shape[1])
          input_img=input_img.type(torch.FloatTensor).to(self.device)
    
          output=self.model(input_img)
          ZFinal[idx]=ptu.to_numpy(output)

        currModel.train()
        return ZFinal
    
    
  def autoenc_logs(self,train_str,val_str,mode="w"):
    with open(self.log_file,mode) as file1:
      file1.write(train_str)
      file1.write("\n")
      file1.write(val_str)
      file1.write("\n")