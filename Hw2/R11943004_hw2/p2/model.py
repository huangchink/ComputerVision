import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2) 
        
        self.fc1 = nn.Linear(64 * 4 * 4, 100) 
        self.fc2 = nn.Linear(100, 10) 
        
        self.relu = nn.ReLU() 

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 4 * 4) 
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
       # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = Identity()
        num_classifier_feature = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_classifier_feature, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )       
        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = MyNet()
    model_resnet18=ResNet18()
    num_params_mynet = sum(p.numel() for p in model.parameters())
    num_params_resnet18 = sum(p.numel() for p in model_resnet18.parameters())
    print('MyNet architecture\n',model)

    print(f'Number of parameters in MyNet: {num_params_mynet}')
    print('Resnet18 architecture\n',model_resnet18)

    print(f'Number of parameters in ResNet18: {num_params_resnet18}')

