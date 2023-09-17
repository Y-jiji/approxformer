import torch as t

from model.approxformer import *
from model.baseline import *
from model.logicformer import *

import lightning as L
import torch.nn.functional as F
from tokenizers import Bert

from typing import *

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

def load_tokenizer(tokenizer: str):
    if tokenizer == 'bert':
        pass
    raise Exception(f"unknown tokenizer name: {tokenizer}")

def main(model: list[str], tokenizer: list[str], dataset: list[str]) -> None:
    import itertools as it
    for model, tokenizer, dataset in it.product(model, tokenizer, dataset):
        # get vocabulary size from tokenizer
        tokenzier = load_tokenizer(tokenizer)
        # get data loader
        pass
        # use pytorch lightning for training
        module = LitWrapper()
        trainer = L.Trainer()
        trainer.fit(module, )
        # run tests
        pass

if __name__ == '__main__':
    import argparse as ap
    import sys
    ap = ap.ArgumentParser()
    ap.add_argument('--model',      nargs='+', required=True)
    ap.add_argument('--dataset',    nargs='+', required=True)
    ap.add_argument('--tokenizer',  nargs='+', required=True)
    args = ap.parse_args(sys.argv[1:])
    print(args.model)