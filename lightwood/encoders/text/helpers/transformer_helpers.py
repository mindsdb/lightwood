"""
2021.03.05

Basic helper functions for PretrainedLang
"""
import torch
from transformers import AdamW

class TextEmbed(torch.utils.data.Dataset):
    """
    Dataset class for quick embedding/label retrieval.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, dataset, device, log=None, optim=None, n_epochs=4):
    """
    Generic training function, given an arbitrary model.
    
    Given a model, train for n_epochs.

    model - torch.nn model;
    dataset - torch.DataLoader; dataset to train
    device - torch.device; cuda/cpu
    log - lightwood.logger.log; print output
    optim - transformers.optimization.AdamW; optimizer
    n_epochs - number of epochs to train

    """
    losses = []
    model.train()
    if optim is None:
        optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(n_epochs):
        for batch in dataset:
            optim.zero_grad() 

            inpids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inpids, attention_mask=attn, labels=labels)
            loss = outputs[0]
            loss.backward()

            optim.step()

        print("Epoch", epoch+1, "Loss", loss)
    return model, losses