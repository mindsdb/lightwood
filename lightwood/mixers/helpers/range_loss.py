import torch




class RangeLoss(torch.nn.Module):
    def __init__(self, confidence_range, reduce='mean', **kwargs):
        super().__init__()
        self.reduce = reduce
        self.confidence_range = confidence_range
        self.log_loss = True
        self.lienar_loss = True

    def forward(self, preds, target):
        target = target.clone()
        for k in range(len(preds)):
            # If the number is within the range desired, apply no penalty (by making the target equall to the prediction)
            if preds[k][0] * (1 + self.confidence_range) > target[k][0] and preds[k][0] * (1 - self.confidence_range) < target[k][0]:
                target[k][0] = preds[k][0]

        loss_arr = []

        if self.lienar_loss:
            loss_arr.append((preds - target) ** 2)
        if self.log_loss:
            loss_arr.append((preds.log() - target.log()) ** 2)

        loss = loss_arr[0]
        for loss_ele in loss_arr[1:]:
            loss += loss_ele
        loss = loss.div(len(loss_ele))

        if self.reduce is False:
            return loss
        if self.reduce == 'mean':
            return torch.mean(loss)

        return torch.mean(loss)
