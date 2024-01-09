from deepfashion.models.embedder.backbones import *

class Embedder(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            in_features: int=64,
            out_features: int = 64,
            normalize: bool = False
            ):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
           nn.ReLU(),
           nn.LazyBatchNorm1d(),
           nn.Linear(in_features, out_features),
           )
        self.normalize = normalize
        
    def forward(self, x):
        y = self.backbone(x)
        y = self.fc(y)
        if self.normalize:
            y = F.normalize(y, p=2, dim=1)
        
        return y

def build_img_embedder(backbone='resnet-18', embed_dim=16):
    if backbone == 'resnet-18':
        backbone_model = ResNet18(out_features = embed_dim)
        in_features = embed_dim
    else:
        raise ValueError(
            ''
        )

    embedder = Embedder(backbone=backbone_model, in_features=in_features,
                        out_features=embed_dim, normalize=False)
    
    return embedder