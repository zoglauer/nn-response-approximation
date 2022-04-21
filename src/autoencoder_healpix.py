from math import inf
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from skimage.util import random_noise
import os
from src.utils import PytorchUtils

ptu = PytorchUtils()


class AutoEncoder(torch.nn.Module):
  
  def __init__(self,cone_model,train_loader,val_loader):
    super().__init__()

    self.cone_model=cone_model
    
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(1728, 800),
      torch.nn.LayerNorm(800),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=0.3),
      torch.nn.Linear(800,450),
      torch.nn.ReLU(),
      torch.nn.Linear(450,250),
      torch.nn.ReLU(),
    )

    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(250,450),
      torch.nn.ReLU(),
      torch.nn.Linear(450,800),
      torch.nn.ReLU(),
      torch.nn.Linear(800,1728),
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
    #super().__init__()


    self.train_dataset=train_loader.dataset
    self.val_dataset=val_loader.dataset
    self.toymodel=toy_model
    self.log_dir=log_dir
    self.log_file=os.path.join(self.log_dir, log_file)
    self.model_file=os.path.join(self.log_dir,"autoenc_model.pth")
    #self.model_file="./autoenc_model.pth"

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
        file1.write("\n")
        num_params=sum([p.numel() for p in self.model.parameters()])
        file1.write("Number of parameters for autoencoder: " + str(num_params))

    
    model_summary()

  def getDataSet(self):

    train_dim1=len(self.train_dataset)*2
    
    val_dim1=len(self.val_dataset)*2
    print("Number of training examples: ", train_dim1)
    print("Number of validation examples: ", val_dim1)
    print()

    train_dim = (train_dim1,) + self.train_dataset[0]['label'].shape
    val_dim = (val_dim1,) + self.val_dataset[0]['label'].shape
    #(val_dim1,val_dim2,val_dim3) , (train_dim1,train_dim2,train_dim3)

    #print("train_dim = ", train_dim)
    #print("val_dim = ", val_dim)
    self.x_train_data = np.zeros(train_dim)
    self.x_val_data = np.zeros(val_dim)
    self.y_train_data = np.zeros(train_dim)
    self.y_val_data = np.zeros(val_dim)
    pos=0
    for i in range(len(self.train_dataset)):

      train_img=self.train_dataset[i]['label']
      #print("Shape of train_img = ", train_img.shape)
      val_img=self.val_dataset[i]['label']
      #print("Shape of val_img = ", val_img.shape)

      noise=['gaussian-1','poisson'] # 'gaussian-2', 'gaussian-3'
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
    print((self.train_dataset[0]['label']).shape[0])
    print((self.val_dataset[0]['label']).shape[0])

    #print(self.x_train_data.shape)
    print()

  
  def tryPlot2D_toyModel(self,Z=None,title="autoencodder_plot_test"):

    XV, YV = np.meshgrid(self.cone_model.gGridCentersXY, self.cone_model.gGridCentersXY)
    if Z is None:
      Z = np.zeros(shape=(self.cone_model.gTrainingGridXY, self.cone_model.gTrainingGridXY))
      Z=self.y_train_data[4]

    fig = plt.figure(0)
    plt.contourf(XV, YV, Z)
    plt.savefig(title)
    plt.close()
    #plt.show()

  
  def add_noise(self,img,noise_type="gaussian"):
    
    #print("Current image shape: ",img.shape)
    origImg=img.copy()
  
    if noise_type=="gaussian-1":
      #print("Adding Gaussian noise")
      noise=np.random.normal(0,0.000001,img.shape)
      img=img+noise
      return (origImg,img)

    if noise_type=="gaussian-2":
      #print("Adding Gaussian noise")
      noise=np.random.normal(0,0.00002,img.shape)
      img=img+noise
      return (origImg,img)


    elif noise_type=="poisson":
      img=random_noise(origImg,mode='poisson')
      return (origImg,img)

    elif noise_type=="speckle":
      noise=np.random.randn(img.shape)
      img=img+img*noise
      return (origImg,img)   
  
  def model_train(self,model):
    epochs=1500
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
        print("Saving model with least loss till now =====>")
        print("#################################################")
        torch.save(self.model.state_dict(), self.model_file)
        self.autoenc_logs(train_str,val_str,"w")
      else:
        if train_epochloss < losslist[-1]:
          #print("Train Loss in epoch {}: {}".format(epoch,train_epochloss))
          print("Least loss till now: ", losslist[-1])
          losslist.append(train_epochloss)
          print("#################################################")
          print("Saving model with least loss till now =====>")
          print("#################################################")
          torch.save(self.model.state_dict(), self.model_file)
          self.autoenc_logs(train_str,val_str,"a")
      print()
      running_loss=0

  def train_epoch(self):
    #print()
    running_loss=0.0
    for input_img,true_img in tqdm((self.trainloader)):
        input_img=input_img.type(torch.FloatTensor)
        true_img=true_img.type(torch.FloatTensor)
        #print("input shape = ", dirty.shape)
        #print()
        #print("ground truth shape = ",clean.shape)
        #print()
        input_img,true_img=input_img.to(self.device),true_img.to(self.device)
    
        #-----------------Forward Pass----------------------
        output=self.model(input_img)
        #print("output shape = ",output.shape)
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
        input_img=input_img.type(torch.FloatTensor)
        true_img=true_img.type(torch.FloatTensor)
        #print(input_img.shape)
        #print()
        #print(true_img.shape)
        #print()
        input_img,true_img=input_img.to(self.device),true_img.to(self.device)
    
        with torch.no_grad():
          output=self.model(input_img)
        #print(output.shape)
        #print()
        loss=self.criterion(output,true_img)

        running_loss+=loss.item()
    self.model.train()
    return running_loss

  def test_model(self,input_imgs=None):
    
    currModel=self.model
    currModel.load_state_dict(torch.load(self.model_file,map_location=torch.device(self.device)))
    currModel.eval()

    if input_imgs==None:
      idx = np.random.randint(0,len(self.valloader.dataset),size=1)
      with torch.no_grad():
        input_img,true_img=self.x_val_data[idx],self.y_val_data[idx]
        input_img=input_img.type(torch.FloatTensor)
        true_img=true_img.type(torch.FloatTensor)

        input_img,true_img=input_img.to(self.device),true_img.to(self.device)

        output=self.model(input_img)
        loss=self.criterion(output,true_img)

        
        print()
        print("Loss for current test image(s) at index {} of test set: {} ".format(idx,loss.item()))
        #print("SNR for given test image: ", 10*np.log(np.mean(output)/np.sqrt(loss.item())))
        print()
        return 
  
    else:
      with torch.no_grad():
        input_img=input_imgs.type(torch.FloatTensor)

        input_img=input_img.to(self.device)

        output=self.model(input_img)
        
    currModel.train()
    return output
  
  def autoenc_logs(self,train_str,val_str,mode="w"):
    with open(self.log_file,mode) as file1:
      file1.write(train_str)
      file1.write("\n")
      file1.write(val_str)
      file1.write("\n")

