import torch

class PARAMS:
    batch_size = 64
    test_batch_size = 64
    epochs = 1
    lr = 1.0
    gamma = 0.7
    no_cuda = True
    seed = 1
    log_interval = 10
    save_model = True

params = PARAMS()
torch.manual_seed(params.seed)
use_cuda = not params.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_kwargs = {'batch_size': params.batch_size}
test_kwargs = {'batch_size': params.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

