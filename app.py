# Areeba fatah
# 21i-0349
# task 2
# A 3


import os
from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app=Flask(__name__)
app.config['UPLOAD_FOLDER']='./static/images/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CIFAR10_CLASSES={
    0:"Airplane",
    1:"Automobile",
    2:"Bird",
    3:"Cat",
    4:"Deer",
    5:"Dog",
    6:"Frog",
    7:"Horse",
    8:"Ship",
    9:"Truck"
}

class PretrainedResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PretrainedResNet,self).__init__()
        self.resnet=models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad=False
        num_features=self.resnet.fc.in_features
        self.resnet.fc=nn.Linear(num_features, num_classes)

    def forward(self,x):
        return self.resnet(x)

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10, image_size=32, patch_size=8, embed_dim=256, num_heads=8, num_layers=12):
        super(VisionTransformer,self).__init__()
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.patch_embedding=nn.Conv2d(3,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.positional_encoding=nn.Parameter(torch.randn(1,(image_size//patch_size)**2,embed_dim))
        encoder_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.fc=nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        x=self.patch_embedding(x).flatten(2).transpose(1,2)
        x+=self.positional_encoding
        x=self.transformer(x)
        x=self.fc(x[:,0])
        return x

class HybridCNNMLP(nn.Module):
    def __init__(self,num_classes=10):
        super(HybridCNNMLP,self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),nn.ReLU(),nn.MaxPool2d(2)
        )
        self.mlp=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,256),nn.ReLU(),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x=self.cnn(x)
        return self.mlp(x)

device=torch.device('cpu')
vit_model=VisionTransformer(num_classes=10).to(device)
cnn_mlp_model=HybridCNNMLP(num_classes=10).to(device)
resnet_model=PretrainedResNet(num_classes=10).to(device)

vit_model.load_state_dict(torch.load('./models/vit_model.pth',map_location=device))
cnn_mlp_model.load_state_dict(torch.load('./models/hybrid_model.pth',map_location=device))
resnet_model.load_state_dict(torch.load('./models/resnet_model.pth',map_location=device))

vit_model.eval()
cnn_mlp_model.eval()
resnet_model.eval()

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return 'No image uploaded!'
    
    image=request.files['image']
    model_choice=request.form['model_choice']

    filepath=os.path.join(app.config['UPLOAD_FOLDER'],image.filename)
    image.save(filepath)

    image_data=Image.open(filepath).convert('RGB')
    input_tensor=transform(image_data).unsqueeze(0).to(device)

    if model_choice=='vit':
        model=vit_model
    elif model_choice=='cnn_mlp':
        model=cnn_mlp_model
    else:
        model=resnet_model

    with torch.no_grad():
        output=model(input_tensor)
        _,predicted=torch.max(output,1)

    predicted_label=CIFAR10_CLASSES[predicted.item()]

    return render_template('result.html',image_path=filepath,label=predicted_label)

if __name__=='__main__':
    app.run(debug=True)
