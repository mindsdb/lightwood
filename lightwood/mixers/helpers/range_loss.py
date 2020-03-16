import torch




class RangeLoss(torch.nn.Module):
    def __init__(self, confidence_range, reduce='mean', **kwargs):
        super().__init__()
        self.reduce = reduce
        self.confidence_range = confidence_range
        self._approximate_sign_data = True
        self._approximate_zero_value = True

    def forward(self, preds, target):
        target = target.clone()

        approximation_indexes = []
        if self._approximate_sign_data:
            approximation_indexes.append(0)
        if self._approximate_zero_value:
            approximation_indexes.append(1)

        for k in range(len(preds)):
            for i in approximation_indexes:
                # For is-zer and is-negative we don't care about the exact value, so just check that it's in the correct ballpark and, if so, apply no penalty (by making the target equll to the prediction)
                if abs(preds[k][i]) < 0.01 and target[k][i] == 0:
                    target[k][i] = preds[k][i]
                if preds[k][i] > 0.99 and preds[k][i] < 1.01 and target[k][i] == 1:
                    target[k][i] = preds[k][i]

            # If 0, we don't care about the number predicted
            if preds[k][2] > 0.9:
                target[k][1] = preds[k][1]
            else:
                # If the number is within the range desired, apply no penalty (by making the target equall to the prediction)
                if preds[k][1] * (1 + self.confidence_range) > target[k][1] and preds[k][1] * (1 - self.confidence_range) < target[k][1]:
                    target[k][1] = preds[k][1]

        loss = (preds - target) ** 2

        if self.reduce is False:
            return loss
        if self.reduce == 'mean':
            return torch.mean(loss)

        return torch.mean(loss)
