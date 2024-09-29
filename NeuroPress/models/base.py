import torch 
import torch.nn as nn 


class Qmodel(nn.Module): 
    def __init__(self,):
        super().__init__()

    def freeze(self): 
        for module in self.modules(): 
            if hasattr(module, 'freeze_layer'):
                module.freeze_layer()
    
    def unfreeze(self): 
        for module in self.modules(): 
            if hasattr(module, 'unfreeze_layer'):
                module.unfreeze_layer()