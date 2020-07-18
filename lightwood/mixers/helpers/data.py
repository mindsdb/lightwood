import torch


def list_collate(batch):
    labels = []
    inputs = []
    for item in batch:
        inputs.append(item[0])
        label = None
        for lb in item[1]:
            if label is None:
                label = lb
            else:
                label = torch.cat((label,lb), 0)
        labels.append(label)

    return [inputs, torch.stack(labels)]
