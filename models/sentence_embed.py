import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
class Sentence_Embed(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.backbone = SentenceTransformer(f'sentence-transformers/{model}')
        self.head = nn.Linear(768, num_classes)
    
    def forward(self, x):
        features_dict = self.backbone.tokenize(x)
        for key in features_dict:
            if isinstance(features_dict[key], torch.Tensor):
                features_dict[key] = features_dict[key].to(self.backbone.device)

        out_features = self.backbone.forward(features_dict)
        embeddings = out_features["sentence_embedding"]

        output = self.head(embeddings)
        return output