import pandas as pd 
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
from os import listdir
from os.path import isdir
import argparse

# defing a function to set parameters for entry in command line
def arg_parser():
    parser=argparse.ArgumentParser(description="settings")
    
     #save directory file  of model
    parser.add_argument('--save_dir',type=str,help='choose an appropriate name of file  to save your trained model')
    
    #adding architecture
    parser.add_argument('--arch',type=str,help='choose any pretrained architecture available from torchvision')
    #adding hidden layers
    parser.add_argument('--hidden_units',type=int,help='select number of hidden layers you want')
    
    #adding learning rate
    parser.add_argument('--learning_rate',type=float,help='select a suitable value of learning rate')
    
    #adding epochs
    parser.add_argument('--epochs',type=int,help='add number of epochs you want reduce the error')
    
    #enable gpu
    parser.add_argument('--gpu',action='store_true',help='use gpu to tran ypur model')
    
    args=parser.parse_args()
    return args

#functions for train transform and loading training data
def train_transform(train_dir):
    train_transforms=transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    return train_data

#function for test/validation transforms and loading testing data
def test_transform(test_dir):
    test_transforms=transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms)
    return test_data

#defining training loader
def train_loader(train_data):
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    return trainloader

#defining test/validation loader
def test_loader(test_data):
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64)
    return testloader

#function to check if gpu arguement is true or not
def gpu_on(gpu_arg):
    if not gpu_arg:
        return torch.device('cpu')
   
      #checking for availability of cuda
    if torch.cuda.is_available():
            device=torch.device("cuda:0")
    else:
            device=torch.device("cpu")
            print("cuda is not available")
            
    return device
#loading the model form tirch vision,by default we load vgg11
def create_model(arch="vgg11"):
        if type(arch)==type(None):
            model=models.vgg11(pretrained=True)
            model.name="vgg11"
        else:
            model = eval("models.{}(pretrained=True)".format(arch))
            model.name = arch
        
        print(f"model selected is {model.name}")
        #freezing parameters
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
 #creating a classifier for our pretrained model
def create_classifier(model,hidden_layers):
    if type(hidden_layers)==type(None):
       hidden_layers=512
 #defining input layers
    if(model.name=="alexnet"):  #for alexnet
        input_features=9216
    elif(model.name=="densenet121"):
        input_features=model.classifier.in_features  #for densenet
    else:
        input_features=model.classifier[0].in_features  #for vgg 
  #define classifer
    model.classifier=nn.Sequential(nn.Linear(input_features,hidden_layers),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_layers,102),
                                   nn.LogSoftmax(dim=1))
    return model

#training the network 
def training(model,trainloader,validloader,device,
             criterion,optimizer,epochs,print_every,steps):
       #check for epoch from argumements
    if type(epochs)==type(None):
        epochs=5

            #training process
    for epoch in range(epochs):
                rloss=0
                for images, labels in trainloader:
                    steps += 1
                    # Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()

                    logps = model.forward(images)#forward pass
                    loss = criterion(logps, labels)
                    loss.backward()#backward pass
                    optimizer.step() #optimizing step using adam

                    rloss += loss.item()#summing up runnning loss

                    if steps % print_every == 0:
                        tloss = 0
                        accuracy = 0
                        model.eval()             #for validation purpose we disable dropout
                        with torch.no_grad():   # turning off gradients for evaluation to reduce computations
                            for images, labels in validloader:    #validation begins
                                images, labels = images.to(device), labels.to(device)
                                logps = model.forward(images)
                                bloss = criterion(logps, labels)

                                tloss += bloss.item()

                                ps = torch.exp(logps)
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        print(f"Epoch {epoch+1}/{epochs}.. "             #printing results based on each epoch
                           f"Train loss: {rloss/print_every:.3f}.. "
                           f"Validation loss: {tloss/len(validloader):.3f}.. "
                           f"Validation accuracy: {accuracy/len(validloader):.3f}")
                        rloss = 0
                        model.train()
    return model
    

 #testing model
def test(model,testloader,device):
    accuracy=0
    with torch.no_grad():
            for inputs,labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {(accuracy/len(testloader))*100:.3f}%")
       
#saving initial checkpoint
def save_model(model,save_dir,train_data):

       model.class_to_idx=train_data.class_to_idx
       checkpoint={'model_architecture':model.name,
                  'model_classifier':model.classifier,
                'model_state_dict':model.state_dict(),
                'model_mapping':model.class_to_idx,
              }
    
    #save_directory has been created to save all checkpoints
    #if not available it will save in the default directory
       if type(save_dir)==type(None):
             torch.save(checkpoint,"checkpoint.pth")
             print("saved in your current directory as not specified")
       else:
           if isdir(save_dir):
            save_dir=save_dir+"/checkpoint.pth"
            torch.save(checkpoint,save_dir)
           else:
            torch.save(checkpoint,"checkpoint.pth")
            print("saved in current directory as incorrect director specfied")
          
             
  #main function
def main():
            #gettin arguements for training
             args=arg_parser()
            #setting directory for training
             data_dir = 'flowers'
             train_dir = data_dir + '/train'
             valid_dir = data_dir + '/valid'
             test_dir = data_dir + '/test'
            #setting transforms on them
             train_data=train_transform(train_dir)
             valid_data=test_transform(valid_dir)
             test_data=test_transform(test_dir)
            #setting up data loaders
             trainloader=train_loader(train_data)
             validloader=test_loader(valid_data)
             testloader=test_loader(test_data)
            
             #load model
             model=create_model(arch=args.arch)
             
             #building classfier
             model=create_classifier(model,hidden_layers=args.hidden_units)
             
             #setting the learning rate
             if type(args.learning_rate)==type(None):
                  learning_rate=0.001
             else:
                   learning_rate=args.learning_rate
            #defining loss and optimizer
             criterion=nn.NLLLoss()
             optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)
            
             #gpu settings
             device=gpu_on(gpu_arg=args.gpu)
             
             #enable the model to use gpu
             model.to(device)
             
             #define print_every and steps
             steps=0
             print_every=40        
   
             #training the model
             model=training(model,trainloader,validloader,device,
                            criterion,optimizer,args.epochs,print_every,steps)
             print("model trained")
             print("lets test for accuracy on testing set")
             #testing the model
             test(model,testloader,device)
             
            
             print("Lets create a checkpoint")
             #save model
             save_model(model,args.save_dir,train_data)
            
             print("checkpoint created")
             
#running model
if __name__ == '__main__': main()
             
 
       
       
       
    
       
        
            
        

     
    
    



         


    
    
    
    
    
    

    
    
    
    
    