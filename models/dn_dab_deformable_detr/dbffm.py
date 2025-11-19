import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchFusionBlock(nn.Module):
    def __init__(self, dim, 
                 alpha=0.5, 
                 learnable_alpha=True):
        """
        Args:
            dim: channel dim
            alpha: weight（0~1）
            learnable_alpha: Whether to make alpha learnable
            token_mixer_for_global: global token mixer module
            mixer_kernel_size: token mixer kernel size 
            local_size: token mixer local window
        """
        super(DualBranchFusionBlock, self).__init__()

        # self.branch1 = 
        # self.branch2 = 

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        alpha = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
        return alpha * out1 + (1 - alpha) * out2

        # Ablation study
        # out1 = self.branch1(x)
        # out2 = self.branch2(out1)
        # return out2

        # Ablation study
        # out1 = self.branch2(x)
        # out2 = self.branch1(out1)
        # return out2

        # Ablation study
        # out = self.branch1(x)
        # return out

        # Ablation study
        # out = self.branch1(x)
        # return out
    
# debug
# fusion_block = DualBranchFusionBlock(dim=64, alpha=0.6)

# # fusion_block = DualBranchFusionBlock(dim=64, alpha=0.5, learnable_alpha=True)

# # input
# x = torch.randn(1, 64, 128, 128)

# # output
# y = fusion_block(x)
# print(y.shape)  # torch.Size([1, 64, 128, 128])
