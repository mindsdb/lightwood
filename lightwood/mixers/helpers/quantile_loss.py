import torch


class RangeLoss(torch.nn.Module):
    def __init__(self, reduce='mean', **kwargs):
        super().__init__()
        self.reduce = reduce
        self.range = 0.02
        self.use_log = False

    def forward(self, preds, target):
        target = target.clone()

        for k in range(len(preds)):
            for i in [0,2]:
                # For is-zer and is-negative we don't care about the exact value, so just check that it's in the correct ballpark and, if so, apply no penalty (by making the target equll to the prediction)
                if preds[k][i] < 0.1 and target[k][i] == 0:
                    target[k][i] = preds[k][i]
                if preds[k][i] > 0.9 and target[k][i] == 1:
                    target[k][i] = preds[k][i]

            # If 0, we don't care about the number predicted
            if preds[k][2] > 0.9:
                target[k][1] = preds[k][1]
            else:
                # If the number is within the range desired, apply no penalty (by making the target equall to the prediction)
                if preds[k][1] * (1 + self.range) > target[k][1] and preds[k][1] * (1 - self.range) < target[k][1]:
                    target[k][1] = preds[k][1]

        if self.use_log:
            preds = preds.clone()
            preds[:,1] = preds[:,1].log()
            target[:,1] = target[:,1].log()

        loss = (preds - target) ** 2

        if self.reduce is False:
            return loss
        if self.reduce == 'mean':
            return torch.mean(loss)

        return torch.mean(loss)


class QuantileLoss(torch.nn.Module):
    def __init__(self, reduce='mean', **kwargs):
        super().__init__()
        self.reduce = reduce

    def forward(self, preds, target):
        print(preds)
        main_mse_loss = (preds[:,:3] - target[:,:3]) ** 2

        lowe_range_loss = []
        for i in range(len(preds)):
            if preds[i,3] > preds[i,2]:
                lowe_range_loss.append([ (preds[i,4] - target[i,2]) * 2 ** 2])
            elif preds[i,3] > target[i,2]*0.95:
                lowe_range_loss.append([0])
            else:
                lowe_range_loss.append([ (preds[i,3] - target[i,2]*0.95)  ** 2])

        lowe_range_loss = torch.Tensor(lowe_range_loss).to(preds.device)

        upper_range_loss = []
        for i in range(len(preds)):
            if preds[i,4] < preds[i,2]:
                upper_range_loss.append([ (preds[i,4] - target[i,2]) * 2 ** 2])
            elif preds[i,4] < target[i,2]*1.05:
                upper_range_loss.append([0])
            else:
                upper_range_loss.append([ (preds[i,4] - target[i,2]*1.05) ** 2])

        upper_range_loss = torch.Tensor(upper_range_loss).to(preds.device)

        loss = torch.cat([main_mse_loss, lowe_range_loss, upper_range_loss], 1)

        if self.reduce is False:
            return loss
        if self.reduce == 'mean':
            return torch.mean(loss)

        return torch.mean(loss)

'''
class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles=[0.85,0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i+3]
            losses.append(
                torch.max(
                   (q-1) * errors,
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
'''
