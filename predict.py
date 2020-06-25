import argparse
import torch
import numpy as np
import pandas as pd
import PIL
import json
from torchvision import models  

#deifning argument parser arguements
def arg_parser():
    
    parser=argparse.ArgumentParser(description="settings")
   
   #adding the image
    parser.add_argument('--image',type=str,help='add the path for the image file')
    
    
    #adding checkpoint
    parser.add_argument('--checkpoint',type=str,help='mention the checkpoint file')
    
    #adding the top_k number
    parser.add_argument('--top_k',type=int,help='add the top_k number')
    
    #adding category names
    parser.add_argument('--category_names',type=str,help='add names of categories for the predictions')
   
    #enable gpu
    parser.add_argument('--gpu',action='store_true',help='use gpu to train your model')
    
     # Parse args
    args = parser.parse_args()
    
    return args
    
  #loading the required model
def load_model(path):
    #loading the path
    checkpoint=torch.load(path)
    #print(checkpoint["model_architecture"])
    #loading model
    model = eval("models.{}(pretrained=True)".format(checkpoint['model_architecture']))
   
    #freezing parameters
    for param in model.parameters():
        param.requires_grad=False
    
    model.classifier=checkpoint['model_classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx=checkpoint['model_mapping']
    
    return model
   
 #processing the image
def process_image(image):
    
    im=PIL.Image.open(image)
    width,height=im.size
    
    if width<height:
        new_size=[256,256**600]
    else:
         new_size=[256**600,256]
    
    im.thumbnail(size=new_size)
    
    center=width/4,height/4
    left,top,right,bottom=center[0]-(244/2),center[1]-(244/2),center[0]+(244/2),center[1]+(244/2)
    im=im.crop((left,top,right,bottom))
    np_im=np.array(im)/255
    
    means=[0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    
    np_im=(np_im-means)/std
    
    np_im=np_im.transpose(2,0,1)
    
    return np_im

def predict(image_path, model, topknum=5):
    
    model.to("cpu")
    model.eval()
    
    #process image using process_img function
    
    img=process_image(image_path)
    img_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).type(torch.FloatTensor).to("cpu")
    
    with torch.no_grad():
        output=model.forward(img_tensor)
        
    ps=torch.exp(output)
    top_k,top_labels=ps.topk(topknum)
    
    top_k=top_k.tolist()[0]
    top_labels=top_labels.tolist()[0]
    
    mapping={value: key for key, value in
                model.class_to_idx.items()
            }
    
    classes=[mapping[item] for item in top_labels]
    classes=np.array(classes)
    
    return top_k,classes

#the gpu settings for our program
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
    
 #main function
def main():
       args=arg_parser()
    
     #load catgories to names json file
       with open(args.category_names, 'r') as file:
            cat_to_name=json.load(file)
    
       #load model trained with train.py   
       model=load_model(args.checkpoint)
    
       #check for gpu
       device=gpu_on(gpu_arg=args.gpu)
    
    
    #initialize topknum
    
       if type(args.top_k)==type(None):
            topknum=5
       else:
           topknum=args.top_k 
    #find the top _k values and classes
       
       top_k,classes=predict(args.image, model, topknum=5)
       
       #print predictions for the dataset
        
       class_names=[cat_to_name[item] for item in classes]
    
       #printing predictions
       print("label:={}".format(classes))
       print("probabilities:={}".format(top_k))
       print("class_name:={}".format(class_names))
       print ("Excution over")
       
    
    
if __name__ == '__main__': main()
    
    
    
      
    
