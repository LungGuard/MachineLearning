import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 activation=nn.ReLU(),
                 dropout_p=0.0):  
        super(DenseBlock, self).__init__()

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation
        #self.dropout = nn.Dropout(p=dropout_p) 

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        #x = self.dropout(x)
        return x