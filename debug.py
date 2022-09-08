from models.diffusion import TransformerConcatLinear
import torch
import numpy as np
from torchsummary import summary

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    model = TransformerConcatLinear(point_dim=2, context_dim=512, tf_layer=3, residual=False)
    print(summary(model, (640,480,3)))

if __name__ == '__main__':
    main()
