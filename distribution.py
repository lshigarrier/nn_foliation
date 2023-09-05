import torch
import torch.utils.data
from torch.linalg import eigh, matrix_rank
from fisher_metric import compute_fim
from simplelstm import DynamicalSystemDataset, SimpleLSTM, SimpleLinear
from generate_data import xdot_sin, xdot_linear, load_yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm


class Distribution:

    def __init__(self):

        self.args = load_yaml()

        if self.args['system'] == 'sin':
            self.xdot = xdot_sin
        elif self.args['system'] == 'linear':
            self.xdot = xdot_linear
        
        self.device = torch.device("cuda")

        if self.args['cpu'] or not torch.cuda.is_available():
            self.device = torch.device("cpu")

        if self.args['verbose']:
            print(f"Using device: {self.device}")
            print(f"args cpu: {self.args['cpu']}\ncuda is available: {torch.cuda.is_available()}")
            if self.device.type == 'cuda':
                current = torch.cuda.current_device()
                print(f"cuda device count: {torch.cuda.device_count()}\ncuda current device: {current}\n"
                      f"{torch.cuda.device(current)}\ncurrent device name: {torch.cuda.get_device_name(current)}")
        
        self.dataset = DynamicalSystemDataset(self.args['data_file'], self.args['obs'], self.args['preds'])

        self.model = SimpleLSTM(hid_size=self.args['hid_size'], layers=self.args['layers']).to(self.device)
        # self.model = SimpleLinear(hid_size=self.args['hid_size'], layers=self.args['layers'])
        self.model.load_state_dict(torch.load(f'models/Dynamical_System/{self.args["name"]}/{self.args["model_pth"]}',
                                              map_location=torch.device('cpu')))

        self.dl = torch.utils.data.DataLoader(self.dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=0)

        self.data_mean = self.dataset.data_mean
        self.data_std = self.dataset.data_std

        self.model.eval()

    def traj_pred(self, x, nb_step=48, inp_size=3, recursive=False):
        inp = torch.Tensor(x[:inp_size]).unsqueeze(0).unsqueeze(-1)
        inp = (inp.to(self.device) - self.data_mean.to(self.device)) / self.data_std.to(self.device)
        preds = [inp[0, i, 0] for i in range(inp.shape[1])]
        stds = [0 for _ in range(inp.shape[1])]
        for t in range(nb_step):
            pred = self.model(inp)
            # pred = self.model(inp.squeeze(dim=2))
            preds.append(pred[0, 0])
            if recursive:
                stds.append(stds[-1] + pred[0, 1])
                inp = torch.Tensor([inp[0, 1, 0], inp[0, 2, 0], pred[0, 0]]).unsqueeze(-1).unsqueeze(0)
            else:
                stds.append(pred[0, 1])
                if t < nb_step - 1:
                    inp = torch.Tensor([x[t+1], x[t+2], x[t+3]]).unsqueeze(-1).unsqueeze(0)
                    inp = (inp.to(self.device) - self.data_mean.to(self.device)) / self.data_std.to(self.device)
        preds = np.array([(preds[i]*self.data_std.to(self.device) + self.data_mean.to(self.device)).detach().numpy()[0] for i in range(len(preds))])
        stds = np.array([(stds[i]*self.data_std.to(self.device)).detach().numpy()[0] for i in range(len(stds))])
        return preds, stds
        
    def compute_kernel(self, inp, kernel=False, acc=False):
    
        inp = (inp.to(self.device) - self.data_mean.to(self.device)) / self.data_std.to(self.device)
        inp.requires_grad = True
        
        pred = self.model(inp)

        input_dim = inp.shape[1]*inp.shape[2]
        output_dim = pred.shape[1]
        jac = torch.zeros((inp.shape[0], output_dim, input_dim)).to(self.device)
        pred_flat = pred.contiguous().view(pred.shape[0], -1).to(self.device)
        for i in range(output_dim):
            grd = torch.zeros(pred_flat.shape).to(self.device)
            grd[:, i] = 1
            pred_flat.backward(gradient=grd, retain_graph=True)
            jac[:, i, :] = inp.grad.contiguous().view(inp.shape[0], -1)
            inp.grad.zero_()
            
        g_out = compute_fim(pred.unsqueeze(1), device=self.device, eps=self.args['eps'])
        jac_t = jac.transpose(-2, -1)
        g_in = torch.bmm(jac_t, torch.bmm(g_out, jac))
        
        try:
            values, vectors = eigh(g_in)
        except:
            print(f'Error: {g_in}')
        for i in range(inp.shape[0]):
            order = torch.abs(values[i]).argsort()
            vectors[i] = vectors[i, :, order]
            values[i] = values[i, order]
            
        # if self.args['verbose']:
        #    print(f"g_in:{g_in[0].shape}")
        #    print(f"rank g_in:{matrix_rank(g_in[0])}")
        #    print(f"rank g_out:{matrix_rank(g_out[0])}")
        #    print(f"rank jac:{matrix_rank(jac[0])}")

        if kernel:
            if acc:
                return vectors[:, :, 0], pred
            return vectors[:, :, 0]
        if acc:
            return vectors[:, :, 1:], pred
        return vectors[:, :, 1:]

    def plot_one_traj(self, x0, t, recursive, fig, ax):
        x = np.squeeze(odeint(self.xdot, x0, t))
        preds, stds = self.traj_pred(x, recursive=recursive)
        # print(f'x0:{x0}\nstds:{stds}')
        ax.plot(t, x, c='b')
        ax.plot(t, preds, c='g')
        ax.fill_between(t, preds - stds, preds + stds, color='r', alpha=0.3)
        ax.set_xlim(left=0)
        return fig, ax, x, preds

    def likelihood(self, pred, std, trg):
        std = max(std, self.args['eps'])
        p = norm.cdf(trg, loc=pred, scale=std)
        if trg > pred:
            return 2*(1-p)
        else:
            return 2*p


def test_accuracy(distri, datatype='train'):
    likely_list = []
    accuracy_list = []
    dataset = None
    if datatype == 'train':
        dataset = distri.dataset
    elif datatype == 'val':
        dataset = DynamicalSystemDataset('data/data_xdsin_val.txt', distri.dataset.inp_len, distri.dataset.out_len)
    elif datatype == 'in':
        dataset = DynamicalSystemDataset('data/data_xdsin_in_distri.txt', distri.dataset.inp_len, distri.dataset.out_len)
    elif datatype == 'out':
        dataset = DynamicalSystemDataset('data/data_xdsin_out_distri.txt', distri.dataset.inp_len, distri.dataset.out_len)
    for data in dataset:
        inp, trg = (data['src'] - distri.data_mean.to(distri.device))/distri.data_std.to(distri.device), data['trg']
        out = distri.model(inp.unsqueeze(0)).detach().numpy()
        pred = out[0, 0]*distri.data_std + distri.data_mean
        std = out[0, 1]*distri.data_std
        trg = trg.detach().numpy()[0]
        likely_list.append(distri.likelihood(pred, std, trg))
        accuracy_list.append(abs((pred-trg)/trg))
    print(f'Relative absolute error: {np.array(accuracy_list).mean()*100:.2f} %')
    print(f'Likelihood: {np.array(likely_list).mean()*100:.2f} %')


def test_traj(distri, recursive=False):
    n = len(distri.dataset)
    print(f'Dataset length: {n}')
    dx = distri.args['dx']
    n1, n2, n3 = distri.args['n1'], distri.args['n2'], distri.args['n3']
    xlim = (-distri.args['plotlim'], distri.args['plotlim'])
    # x0s = distri.dataset.data[:, 0]
    x0s = np.linspace(xlim[0] - dx, xlim[1] + 2*dx, n1+n2)
    t = np.linspace(0, 1, n3)
    fig, ax = plt.subplots()
    for x0 in x0s:
        fig, ax, _, _ = distri.plot_one_traj(x0, t, recursive, fig, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(xlim[0] - 2*dx, xlim[1] + 3*dx)
    ax.set_aspect(1/(xlim[1] - xlim[0] + 5*dx))
    plt.show()


def test():
    distri = Distribution()
    for i in range(10):
        data = distri.dataset[i]
        inp = data['src'].unsqueeze(0)
        v, pred = distri.compute_kernel(inp)
        print(f'vectors:{v}')


def main():
    distri = Distribution()
    if distri.args['test'] == 'traj':
        test_traj(distri, recursive=distri.args['recursive'])
    elif distri.args['test'] == 'accuracy':
        test_accuracy(distri, datatype=distri.args['datatype'])


if __name__ == "__main__":
    main()
