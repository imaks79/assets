import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Resnet18_modified(nn.Module):
    def __init__(self, input_channels = 4, n_classes = 2):
        super().__init__();    
        self.Conv2d_new_head = nn.Conv2d(   
                                            in_channels = input_channels, 
                                            out_channels = 3, 
                                            kernel_size = 7, 
                                            stride = 2, 
                                            padding = 3, 
                                            bias = False
                                            );
        self.model = resnet18(weights = ResNet18_Weights.DEFAULT);
        self.model.fc  = nn.Linear(in_features = self.model.fc.in_features, out_features = n_classes); 
        self.apply_weights([self.Conv2d_new_head, self.model.fc]);
        for param in self.model.parameters():
            param.requires_grad = False;
        
        
    def forward(self, x):
        x = self.Conv2d_new_head(x);
        x = self.model(x);
        return x;
        
        
    def apply_weights(self, list_of_layers):
        for m in list_of_layers:
            m.apply(self.init_weights);
            
            
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight);
        elif isinstance(m, nn.Linear): 
            nn.init.xavier_uniform_(m.weight);
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.bias != None:
                m.bias.data.fill_(0);