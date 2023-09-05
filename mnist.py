import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import Lenet, FeatureNet, RobustMnist
from mnist_utils import moving_average, load_yaml
from attacks_utils import FastGradientSignUntargeted
import matplotlib.pyplot as plt
import time
import psutil
import os


def train(param, model, device, train_loader, optimizer, epoch, attack=None):
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if param['adv_train']:
            # When training, the adversarial example is created from a random
            # close point to the original data point. If in evaluation mode,
            # just start from the original data point.
            adv_data = attack.perturb(data, target, True)
            model.eval()
            output = model(adv_data)
            model.train()
        else:
            output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*data.shape[0]
        if param['verbose'] and (batch_idx % param['log_interval'] == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
            if param['adv_train']:
                with torch.no_grad():
                    model.eval()
                    stand_output = model(data)
                    model.train()
                pred = stand_output.argmax(dim=1, keepdim=True)
                std_acc = pred.eq(target.view_as(pred)).float().mean().item() * 100
                pred = output.argmax(dim=1, keepdim=True)
                adv_acc = pred.eq(target.view_as(pred)).float().mean().item() * 100
                print('Standard acc: %.3f %%, Robustness acc: %.3f %%' % (std_acc, adv_acc))
    epoch_loss /= len(train_loader.dataset)
    if param['verbose']:
        print('Train set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss


def test(param, model, device, test_loader, attack=None):
    model.eval()
    test_loss = 0
    correct = 0
    adv_correct = 0
    tic = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            if data.shape[0] == 1:
                pred = output.argmax(dim=1, keepdim=True)[0]
            else:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if param['adv_train']:
                # use predicted label as target label
                # with torch.enable_grad():
                adv_data = attack.perturb(data, pred, False)
                adv_output = model(adv_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                adv_correct += adv_pred.eq(target.view_as(pred)).sum().item()
            if param['verbose'] and (batch_idx % param['log_interval'] == 0):
                print(f'Iteration: {batch_idx}')
                print(f'Elapsed time (s): {time.time() - tic}')
                print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
                tic = time.time()
    test_loss /= len(test_loader.dataset)
    if param['adv_test']:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Robust accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
            adv_correct, len(test_loader.dataset), 100. * adv_correct / len(test_loader.dataset)))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def init_model(param, device, name, weights):
    model = Lenet(param).to(device)
    model.load_state_dict(torch.load(f'models/mnist/{name}/{weights}', map_location='cpu'))
    model.eval()
    return model


def initialize(param, device):
    if param['load']:
        test_kwargs = {'batch_size': param['attack_batch_size']}
        print(f'Using attack batch size for test loader')
    else:
        test_kwargs = {'batch_size': param['test_batch_size']}
        print(f'Using standard batch size for test loader')
    if param['robust_data']:
        train_kwargs = {'batch_size': param['robust_batch_size']}
        print(f'Using robust batch size for train loader')
    else:
        train_kwargs = {'batch_size': param['batch_size']}
        print(f'Using standard batch size for train loader')
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if param['robust_mnist']:
        dataset1 = RobustMnist(param)
        print(f'Using robust MNIST dataset')
    else:
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())
        print(f'Using standard MNIST dataset')
    dataset2 = datasets.MNIST('./data/mnist', train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    if param['robust_data']:
        model = FeatureNet(param).to(device)
        print(f'Using feature model')
    else:
        model = Lenet(param).to(device)
        print(f'Using standard model')
    if param['load']:
        print(f'Loading weights')
        model.load_state_dict(torch.load(f'models/mnist/{param["name"]}/{param["model"]}', map_location='cpu'))
        model.eval()
    else:
        print(f'Randomly initialized weights')

    print('Initialization done')

    return train_loader, test_loader, model


def training(param, device, train_loader, test_loader, model, attack=None):
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss_list = []
    for epoch in range(1, param['epochs'] + 1):
        epoch_loss = train(param, model, device, train_loader, optimizer, epoch, attack=attack)
        test(param, model, device, test_loader, attack=attack)
        if epoch % param['save_step'] == 0:
            torch.save(model.state_dict(), f'models/mnist/{param["name"]}/{epoch:05d}.pt')
        loss_list.append(epoch_loss)
    plt.plot(moving_average(loss_list, 50)[50:])


def main():
    param = load_yaml()
    torch.manual_seed(param['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    train_loader, test_loader, model = initialize(param, device)
    attack = None
    if param['adv_train']:
        attack = FastGradientSignUntargeted(model, device, param['epsilon'], param['alpha'],
                                            min_val=0, max_val=1, max_iters=param['max_iter'], _type=param['perturbation_type'])
    if param['train']:
        training(param, device, train_loader, test_loader, model, attack=attack)
    else:
        test(param, model, device, test_loader, attack=attack)


if __name__ == '__main__':
    main()
