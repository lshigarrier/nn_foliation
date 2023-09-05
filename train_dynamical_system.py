import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from simplelstm import DynamicalSystemDataset, SimpleLSTM, SimpleLinear
from generate_data import load_yaml
import matplotlib.pyplot as plt


def moving_average(array, window_size):
    i = 0
    moving_averages = []
    while i < len(array) - window_size + 1:
        this_window = array[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def main():
    args = load_yaml()

    device = torch.device("cuda")

    if args['cpu'] or not torch.cuda.is_available():
        device = torch.device("cpu")

    if args['verbose']:
        print(f"Using device: {device}")
        print(f"args cpu: {args['cpu']}\ncuda is available: {torch.cuda.is_available()}")
        if device.type == 'cuda':
            current = torch.cuda.current_device()
            print(f"cuda device count: {torch.cuda.device_count()}\ncuda current device: {current}\n"
                  f"{torch.cuda.device(current)}\ncurrent device name: {torch.cuda.get_device_name(current)}")

    train_dataset = DynamicalSystemDataset(args['data_file'], args['obs'], args['preds'])

    model = SimpleLSTM(hid_size=args['hid_size'], layers=args['layers']).to(device)
    # model = SimpleLinear(hid_size=args['hid_size'], layers=args['layers'])
    if args['resume_train']:
        model.load_state_dict(torch.load(f'models/Dynamical_System/{args["name"]}/{args["model_pth"]}'))

    tr_dl = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0)

    optim = Adam(model.parameters())
    epoch = 0
    
    data_mean = train_dataset.data_mean
    data_std = train_dataset.data_std

    loss_list = []
    while epoch < args['max_epoch']:
        epoch_loss = 0
        model.train()

        for id_b, batch in enumerate(tr_dl):

            optim.zero_grad()
            inp = (batch['src'].to(device) - data_mean.to(device))/data_std.to(device)
            pred = model(inp)
            loss = F.gaussian_nll_loss(pred[:, 0], (batch['trg'].to(device) - data_mean.to(device))/data_std.to(device),
                                       torch.square(pred[:, 1]), reduction='mean', eps=args['eps'])
            loss.backward()
            optim.step()
            # if args['verbose']:
            #     print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f"
            #      % (epoch, args['max_epoch'], id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()

        if epoch % args['save_step'] == 0:

            torch.save(model.state_dict(), f'models/Dynamical_System/{args["name"]}/{epoch:05d}.pth')

        if args['verbose']:
            print("train epoch %03i/%03i  loss: %7.4f"
                  % (epoch, args['max_epoch'], epoch_loss/len(train_dataset)))
        loss_list.append(epoch_loss)
        epoch += 1

    plt.plot(moving_average(loss_list, 50)[50:])
    plt.show()


if __name__ == '__main__':
    main()
