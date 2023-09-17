import torch as t

from model.approxformer import *
from model.baseline import *
from model.logicformer import *

import lightning as L
import torch.nn.functional as F

class LitWrapper(L.LightningModule):
    def __init__(self, model: t.nn.Module) -> None:
        super().__init__()
        self.inner = model
    def forward(self, x):
        return self.inner(x)
    def training_step(self, batch):
        x, y = batch
        loss = F.cross_entropy(self.inner(x).log(), y)
        self.log("training_loss", loss.item())
        return loss
    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters())
        return optimizer

def main(
    model='approxformer'|'gpt2', 
    tokenizer='bert'|'gpt2', 
    dataset='wikitext2'|'shakespeare'|'wikitext103'
) -> None:
    module = LitWrapper(
        ApproxFormer() if model == 'approxformer' else
        None
    )
    pass

if __name__ == '__main__':
    pass