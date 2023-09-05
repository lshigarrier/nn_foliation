import torch


def compute_fim(y, device, eps=float('-inf')):
    """
    Compute the Fisher Information Metric (FIM) of the output space at the point y, assuming diagonal covariance matrix

    Input
        y: Output sequence of shape (batch size, sequence length, 2) where the last dimension is (mean, std)
        device: The device of y
        
    Output
        Tensor of shape (batch size, sequence length * 2, sequence length * 2). The coordinate system is (mean, std)
    """
    y = torch.maximum(y[:, :, 1], torch.Tensor([eps]))
    y = torch.square(y)
    temp = torch.ones(y.shape).to(device)
    y = torch.div(temp, y)
    y = y.repeat_interleave(2, dim=1)
    temp = torch.ones(y.shape[0], 2).to(device)
    temp[:, 1] = 2
    temp = temp.repeat(1, y.shape[1]//2)
    return torch.diag_embed(y*temp)


def test():
    y = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                      [[7, 8], [9, 10], [11, 12]]])
    fim = compute_fim(y, 'cpu')
    print(y.shape)
    print(y)
    print(fim)
    print(fim.shape)


if __name__ == '__main__':
    test()
