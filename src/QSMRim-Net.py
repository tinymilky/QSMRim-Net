import torch
import numpy as np
import torch.nn as nn

from backbones.resnet import resnet18

class Swish(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, i):
        result = i * nn.Sigmoid()(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = nn.Sigmoid()(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class SMOTELayer(nn.Module):
    
    def __init__(self):
        
        super(SMOTELayer, self).__init__()
        
        self.fea_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),            
        ) 
        
    def knn(self, x, k):
        
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (bs, num_points, k)
        
        return idx.squeeze(0)

    def forward(self, fea, lbl):
        '''
        fea: (bs, c)
        lbl: (bs, 1)
        '''
        fea = self.fea_transform(fea)
        
        if self.training:
            pos = lbl.squeeze(-1)>0
            if pos.sum()>2:
                knn_fea = fea[lbl.squeeze(-1)>0]
                knn_fea = knn_fea.transpose(1,0).unsqueeze(0) # (1, c, bs)

                top_idxs = self.knn(knn_fea, 3)
                new_fea = []
                for i in range(top_idxs.shape[0]):
                    for j in range(top_idxs.shape[1]):
                        idx_1 = top_idxs[i][0]
                        idx_2 = top_idxs[i][j]
                        lerp = torch.lerp(knn_fea[:,:,idx_1], knn_fea[:,:,idx_2], np.random.rand())
                        new_fea.append(lerp)
                
                new_fea = torch.cat(new_fea, dim=0)
                fea = torch.cat([fea,new_fea], dim=0)
                lbl = torch.cat([lbl, torch.ones(len(new_fea),1).to(fea.device)], dim=0)

        return fea, lbl

class res18L4TwoRadsPlusSMOTENew(nn.Module):

    def __init__(self, in_channels = 2):
        
        super(res18L4TwoRadsPlusSMOTENew, self).__init__()

        self.encoder = resnet18(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.encode_rad = nn.Sequential(
            nn.Linear(527, 1024),
            nn.BatchNorm1d(1024),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            Swish_Module(),
            nn.Dropout(),
            nn.Linear(128, 1),
        )
        
        self.smote = SMOTELayer()

    # Please note that lbl will be used only during training 
    def forward(self, x, rad, lbl):
        
        x = self.encoder(x[:,1:3,...])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        rads = self.encode_rad(rad)
        
        x = x + rads
        x, lbl = self.smote(x, lbl) 
        x = self.classifier(x)
        
        return x, lbl