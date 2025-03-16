
import torch
import torcheval

class StdDev(torcheval.metrics.Metric[torch.Tensor]):
    """
    Standard deviation metric. Implementation of: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, device=None):
        super().__init__(device=device)
        self._add_state("n", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        self._add_state("avg", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        self._add_state("mse", torch.tensor(0.0, device=self.device, dtype=torch.float64))

    def reset(self):
        self.n = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        self.avg = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        self.mse = torch.tensor(0.0, device=self.device, dtype=torch.float64)

    @torch.inference_mode()
    def compute(self):
        if self.n > 1:
            return torch.sqrt(self.mse / (self.n - 1))
        else:
            return self.n

    @torch.inference_mode()
    def update(self, input):
        if input.dim() == 0:
            input = input.unsqueeze(0)
        old_n = self.n
        new_n = input.size(0)

        if new_n >= 1:
            old_avg = self.avg
            new_avg = torch.mean(input)

            self.n += new_n
            self.avg = (old_n * old_avg + new_n * new_avg) / self.n

            delta = new_avg - old_avg

            new_mse = torch.sum((input - new_avg) ** 2)
            self.mse += new_mse + delta ** 2 * old_n * new_n / self.n

        return self

    @torch.inference_mode()
    def merge_state(self, metrics):
        for metric in metrics:
            if metric.inputs:
                self.inputs.append(
                    torch.cat(metric.inputs, dim=metric.dim).to(self.device)
                )
        return self


def grad_norm(parameters):
    """
    Compute the norm of gradients, following: https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
    Call this AFTER backward(). If you are trying to avoid updating with NaN gradients,
    call optimizer.step() AFTER checking this.
    :param parameters: List of model parameters.
    :return: The gradient norm.
    """
    total_norm = torch.zeros(1)

    if len(parameters) > 0:
        device = parameters[0].device
        tmp = [torch.norm(p.grad.detach(), 2).to(device) for p in parameters if p.grad is not None]
        if len(tmp) > 0:
            total_norm = torch.norm(torch.stack(tmp),2.0)

    return total_norm

def avg_accuracy(acc_matrix):
    """
    Compute the average accuracy (across every task, past and future) of the most recent model, ignoring NaN values.
    :param acc_matrix: Square accuracy matrix indexed by [last_trained_task, eval_task].
    :return: Average accuracy score.
    """

    return torch.nanmean(acc_matrix[-1, :]).item()


def avg_forgetting(acc_matrix):
    """
    Compute the average forgetting of the most recent model, ignoring NaN values.
    :param acc_matrix: Square accuracy matrix indexed by [last_trained_task, eval_task].
    :return: Average forgetting score.
        """
    if acc_matrix.shape[0] == 1:
        return 0.
    else:
        acc_star, _ = torch.max(acc_matrix[0:-1, 0:-1], dim=0)  # Discard last row and last column.
        acc = acc_matrix[-1, 0:-1]  # Last row, discarding the last column.
        forg = torch.nanmean(acc_star - acc).item()

        return forg


def backward_transfer(acc_matrix):
    """
    Compute the backward transfer of the most recent model. NaN values are NOT filtered out.
    :param acc_matrix: Square accuracy matrix indexed by [last_trained_task, eval_task].
    :return: Backward transfer score.
    """
    if acc_matrix.shape[0] == 1:
        return 0.
    else:
        ccm_diff = acc_matrix - torch.diag(acc_matrix)
        bwt = (2. * torch.sum(torch.tril(ccm_diff, diagonal=-1))) / float(acc_matrix.shape[0] *
                                                                          (acc_matrix.shape[0] - 1))
        return max(bwt.item(), 0.)


def forward_transfer(acc_matrix):
    """
    Compute the forward transfer of the most recent model. NaN values are NOT filtered out.
    :param acc_matrix: Square accuracy matrix indexed by [last_trained_task, eval_task].
    :return: Forward transfer score.
    """
    if acc_matrix.shape[0] == 1:
        return 0.
    else:
        fwd = (2. * torch.sum(torch.triu(acc_matrix, diagonal=1))) / float(acc_matrix.shape[0] *
                                                                           (acc_matrix.shape[0] - 1))
        return fwd.item()