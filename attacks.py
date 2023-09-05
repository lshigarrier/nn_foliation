import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth
from mnist import initialize, init_model
from mnist_utils import load_yaml
from attacks_utils import fgsm_attack, jacobian, projection, fim_pullback, eig, compute_angle, l2_distance, fim_distance, pgd, robust_loss, FastGradientSignUntargeted, robust_stop
from attacks_vis import plot_accuracy, plot_examples, plot_one_img, plot_one_graph, plot_graph, plot_img, plot_path, plot_hist
import time
import psutil
import os


def horizontal_path(param, src, dest, step, max_iter, model, device):
    """
    :return:
    path along the data leaf from src to the projection of dest on the data leaf
    list of distances (losses) from the current iteration point and dest
    list of entropy for each iteration
    list of prediction for each iteration
    list of probabilities for each iteration
    """
    x = src.flatten()
    y = dest.flatten()

    path = []
    loss_list = []

    ent_list = []
    pred_list = []
    prob_list = []

    for t in range(max_iter):
        data = torch.from_numpy(x).reshape((1, 1, 28, 28)).to(device)
        jac, entropy, pred, prob = jacobian(model, data, device)

        ent_list.append(entropy.squeeze().detach().numpy())
        pred_list.append(pred.squeeze().detach().numpy())
        prob_list.append(prob.squeeze().detach().numpy())

        proj = projection(y - x, jac)
        x += step*proj/np.linalg.norm(proj)

        path.append(x.copy().reshape(28, 28))

        distance = np.linalg.norm(y - x)
        loss_list.append(distance)

        if t % param['log_interval_path'] == 0:
            print(f'Iteration {t}')

    return path, loss_list, ent_list, pred_list, np.array(prob_list)


def horizontal_path_light(param, src, dest, model, device):
    """
    :return:
    path along the data leaf from src to the projection of dest on the data leaf
    list of distances (losses) from the current iteration point and dest
    list of prediction for each iteration
    """
    x = src.flatten()
    y = dest.flatten()
    path = []
    loss_list = []
    pred_list = []
    t = 1
    tic = time.time()
    while (t <= param['max_iter_path']) and (robust_stop(loss_list, t) > param['tol']):
        data = torch.from_numpy(x).reshape((1, 1, 28, 28)).to(device)
        jac, entropy, pred, prob = jacobian(model, data, device)
        pred_list.append(pred.squeeze().detach().numpy())
        proj = projection(y - x, jac)
        x += param['step']*proj/np.linalg.norm(proj)
        path.append(x.copy().reshape(28, 28))
        distance = np.linalg.norm(y - x)
        loss_list.append(distance)
        if t % param['log_interval_path'] == 0:
            print(f'Iteration: {t}')
        t += 1
    print(f'Nb of iterations: {t-1}')
    print(f'Elapsed time (s): {time.time() - tic}')
    print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
    data = torch.from_numpy(y).reshape((1, 1, 28, 28)).to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    pred_list.append(pred.squeeze().detach().numpy())
    path.append(y.copy().reshape(28, 28))
    return path, loss_list, pred_list


def eigencurves(param, model, device, src):
    m = 9
    step = param['step']
    paths = [[] for _ in range(m)]
    dist_lists = [[] for _ in range(m)]
    ent_lists = [[] for _ in range(m)]
    pred_lists = [[] for _ in range(m)]
    prob_lists = [[] for _ in range(m)]
    previous = []

    x0 = src.detach().cpu().numpy().flatten()
    data = torch.from_numpy(x0).reshape((1, 1, 28, 28)).to(device)
    gx, entropy, pred, prob = fim_pullback(model, data, device)
    values, vectors = eig(gx, m)
    for j in range(m):
        direction = vectors[:, j]/np.linalg.norm(vectors[:, j])
        x = x0 - step*direction  # minus sign!
        previous.append(direction)
        paths[j].append(x0.copy().reshape(28, 28))
        paths[j].append(x.copy().reshape(28, 28))
        dist_lists[j].append(0)
        dist_lists[j].append(np.linalg.norm(x-x0))
        ent_lists[j].append(entropy)
        pred_lists[j].append(pred)
        prob_lists[j].append(prob)

    for i in range(param['max_iter_path']):
        if i % param['log_interval_path'] == 0:
            print(f'Iteration {i}')
        for j in range(m):
            data = torch.from_numpy(paths[j][-1]).float().reshape((1, 1, 28, 28)).to(device)
            gx, entropy, pred, prob = fim_pullback(model, data, device)
            values, vectors = eig(gx, m)
            '''
            dot_prod = np.dot(vectors.T, previous[j])
            idx = np.argmax(np.abs(dot_prod))
            direction = vectors[:, idx]/np.linalg.norm(vectors[:, idx])
            if np.linalg.norm(x-x0) < np.linalg.norm(paths[j][-1].flatten()-x0):
            '''
            direction = vectors[:, j]/np.linalg.norm(vectors[:, j])
            if np.dot(direction, previous[j]) < 0:
                previous[j] = -direction.copy()
            else:
                previous[j] = direction.copy()
            x = paths[j][-1].flatten() - step*previous[j]
            paths[j].append(x.copy().reshape(28, 28))
            dist_lists[j].append(np.linalg.norm(x - x0))
            ent_lists[j].append(entropy)
            pred_lists[j].append(pred)
            prob_lists[j].append(prob)

    return paths, dist_lists, ent_lists, pred_lists, np.array(prob_lists)


def robust_dataset(param, model, device, train_loader):
    new_dataset = []
    new_target = []
    t = 0
    dataset = train_loader.dataset
    n_samples = len(dataset)
    tic = time.time()
    for data, target in train_loader:
        data = data.to(device)
        # Draw the starting point of the optimization
        random_index = int(np.random.random()*n_samples)
        x = dataset[random_index][0].unsqueeze(0)
        x = x.to(device)
        # PGD
        x_opt = pgd(param, x, data, model)
        new_dataset.append(x_opt.cpu().detach().squeeze().numpy())
        new_target.append(target.numpy()[0])
        if t % 100 == 0:
            print(f'Iteration: {t}')
            print(f'Elapsed time (s): {time.time() - tic}')
            print(f'Memory usage (GB): {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)}')
            tic = time.time()
        t += 1
        if t % 1000 == 0:
            new_dataset = np.array(new_dataset)
            new_target = np.array(new_target)
            print(new_dataset.shape)
            print(new_target.shape)
            np.save(param['robust_file']+f'_{t}', new_dataset)
            np.save(param['target_file']+f'_{t}', new_target)
            new_dataset = []
            new_target = []
    if new_dataset:
        new_dataset = np.array(new_dataset)
        new_target = np.array(new_target)
        print(new_dataset.shape)
        print(new_target.shape)
        np.save(param['robust_file'], new_dataset)
        np.save(param['target_file'], new_target)


def create_adv_ex(param, model, device, test_loader):
    """
    Create two .npy files:
    - one file containing nb_adv images from test_loader
    - one file containing adversarial examples of these same images (using PGD)
    """
    attack = FastGradientSignUntargeted(model, device, param['epsilon'], param['alpha'],
                                        min_val=0, max_val=1, max_iters=param['max_iter'],
                                        _type=param['perturbation_type'])
    nb_adv = param['nb_adv']
    original = np.zeros((nb_adv, 28, 28))
    adv_ex = np.zeros((nb_adv, 28, 28))
    for idx, (data, target) in enumerate(test_loader):
        if idx >= nb_adv:
            break
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)[0]
        adv_data = attack.perturb(data, pred, False)
        original[idx] = data.squeeze().detach().numpy()
        adv_ex[idx] = adv_data.squeeze().detach().numpy()
    np.save(param['org_file']+param['name'], original)
    np.save(param['adv_file']+param['name'], adv_ex)


# ---------------------------------------------------- Tests -----------------------------------------------------------


def test_attack(param, model, device, test_loader, epsilon, one_ex=False):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_prob, init_pred = output.max(dim=1, keepdim=True)  # get the index of the max log-probability
        init_prob = torch.mul(torch.exp(init_prob), 100)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_prob, final_pred = output.max(dim=1, keepdim=True)  # get the index of the max log-probability
        final_prob = torch.mul(torch.exp(final_prob), 100)
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < param['examples']):
                orig_ex = data.squeeze().detach().cpu().numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_prob.item(), init_pred.item(), orig_ex, final_prob.item(), final_pred.item(), adv_ex))
                if one_ex:
                    break
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < param['examples']:
                orig_ex = data.squeeze().detach().cpu().numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_prob.item(), init_pred.item(), orig_ex, final_prob.item(), final_pred.item(), adv_ex))
                if one_ex:
                    break

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def run_tests(param, model, device, test_loader, epsilons):
    accuracies = []
    examples = []
    for eps in epsilons:
        acc, ex = test_attack(param, model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    return accuracies, examples


def plot_attacks(param, model, test_loader, eps, device):
    accs, exs = run_tests(param, model, device, test_loader, eps)
    fig1 = plot_accuracy(eps, accs)
    fig2 = plot_examples(eps, exs)
    return fig1, fig2


def test_jac(model, test_loader, device):
    data, target = next(iter(test_loader))
    jac = jacobian(model, data, device)
    basis = orth(jac.T).T
    print(np.dot(basis, basis.T))


def test_horizontal(param, model, test_loader, device):
    acc, exs = test_attack(param, model, device, test_loader, param['one_eps'], one_ex=True)
    orig_prob, orig_pred, orig_ex, adv_prob, adv_pred, adv_ex = exs[0]
    path, loss_list, ent_list, pred_list, prob_list = horizontal_path(param, orig_ex, adv_ex, param['step'], param['max_iter_path'], model, device)

    proj = path[-1]
    index = ent_list.index(max(ent_list)) - 2
    print(f'Index: {index}, {pred_list[index]} -> {pred_list[index+1]}')
    anti_adv = path[index]

    x = orig_ex.flatten()
    y = adv_ex.flatten()
    z = proj.flatten()
    anti = anti_adv.flatten()

    output = model(torch.from_numpy(proj).reshape(1, 1, 28, 28).to(device))
    proj_prob, proj_pred = output.max(dim=1, keepdim=True)
    proj_prob = torch.mul(torch.exp(proj_prob), 100).item()
    proj_pred = proj_pred.item()

    out_anti = model(torch.from_numpy(anti_adv).reshape(1, 1, 28, 28).to(device))
    anti_prob, anti_pred = out_anti.max(dim=1, keepdim=True)
    anti_prob = torch.mul(torch.exp(anti_prob), 100).item()
    anti_pred = anti_pred.item()

    fig00 = plot_one_graph(range(param['max_iter_path']), loss_list,
                           "Loss curve", "Iteration", "Distance")
    fig01 = plot_one_graph(range(param['max_iter_path']), ent_list,
                           "Entropy along horizontal path", "Iteration", "Entropy")
    fig02 = plot_one_graph(range(param['max_iter_path']), pred_list,
                           "Prediction along horizontal path", "Iteration", "Prediction")
    fig03 = plot_one_graph(range(param['max_iter_path']), prob_list,
                           "Probabilities along horizontal path", "Iteration", "Probabilities", multiple=True)

    print(f'Origin: {orig_pred}({orig_prob:.2f})\nDestination: {adv_pred}({adv_prob:.2f})\nProjection: {proj_pred}({proj_prob:.2f})')
    print(f'Anti-Adversarial: {anti_pred}({anti_prob:.2f})')

    # angle_proj_dest = compute_angle(x, y, model, device)
    # angle_org_dest = compute_angle(z, y, model, device)

    # print(f'Distance org-dest: l2={np.linalg.norm(y - x)}, inf={np.linalg.norm(y - x, np.inf)}')
    # print(f'Distance org-proj: l2={np.linalg.norm(z - x)}, inf={np.linalg.norm(z - x, np.inf)}')
    # print(f'Distance dest-proj: l2={np.linalg.norm(z - y)}\nAngle dest-proj: {angle_proj_dest}\nAngle dest-org: {angle_org_dest}')

    fig10 = plot_one_img(orig_ex, f'Origin\nPrediction: {orig_pred}, Proba: {orig_prob:.2f}%')
    # fig11 = plot_one_img(adv_ex, f'Destination\nPrediction: {adv_pred}, Proba: {adv_prob:.2f}%')
    fig12 = plot_one_img(proj, f'Projection\nPrediction: {proj_pred}, Proba: {proj_prob:.2f}%, Distance to origin: l2={np.linalg.norm(z-x):.2f}')
    fig13 = plot_one_img(anti_adv, f'Anti-Adversarial Example\nPrediction: {anti_pred}, Proba: {anti_prob:.2f}%, Distance to origin: l2={np.linalg.norm(anti-x):.2f}')

    figs = (fig00, fig01, fig02, fig03, fig10, fig12, fig13)
    return figs


def test_eigencurves(param, model, test_loader, device):
    dataset = test_loader.dataset
    src = dataset[0][0]
    paths, dist_lists, ent_lists, pred_lists, prob_lists = eigencurves(param, model, device, src)
    xdata = range(param['max_iter_path']+1)

    fig0 = plot_graph(xdata, ent_lists,
                      "Entropy along eingendirections", "Iteration", "Entropy")
    fig1 = plot_graph(range(param['max_iter_path']+2), dist_lists,
                      "Euclidean distance along eingendirections", "Iteration", "Distance")
    fig2 = plot_graph(xdata, pred_lists,
                      "Prediction along eingendirections", "Iteration", "Prediction")
    fig3 = plot_graph(xdata, prob_lists,
                      "Probabilities along eingendirections", "Iteration", "Probabilities", multiple=True)
    fig4 = plot_img(paths, ent_lists, pred_lists, prob_lists)

    figs = (fig0, fig1, fig2, fig3, fig4)
    return figs


def test_l2_distance(loader):
    avg, tot_avg, tot_eff = l2_distance(loader)
    print('Euclidean distance')
    print(f'Sample size: {tot_eff}\naverage per class: {avg}\ntotal average: {tot_avg}\n')
    for i, row in enumerate(avg):
        print(f'current class: {i} ; closest class: {np.argmin(row)}')
        print(f'distance with itself: {row[i]} ; min distance: {np.min(row)}\n')


def test_fim_distance(loader, model):
    avg, tot_avg, tot_eff = fim_distance(loader, model)
    print('Riemannian distance')
    print(f'Sample size {tot_eff}\naverage per class: {avg}\ntotal average: {tot_avg}\n')
    for i, row in enumerate(avg):
        print(f'current class: {i} ; closest class: {np.argmin(row)}')
        print(f'distance with itself: {row[i]} ; min distance: {np.min(row)}\n')


def test_horizontal_light(param, model, device, idx):
    images = np.load(param['org_file']+param['name']+'.npy')
    adv_img = np.load(param['adv_file']+param['name']+'.npy')
    # 0 and 11 of the train sample are 5
    src = images[idx]
    dest = images[idx+1]  # adv_img[idx]
    path, loss_list, pred_list = horizontal_path_light(param, src, dest, model, device)
    fig1 = plot_path(path, 7, pred_list)
    fig2 = plot_one_graph(range(len(loss_list)), loss_list, "Loss curve", "Iteration", "Distance")
    return fig1, fig2


def test_leaves(param, device):
    figs = []
    nb_img = param['nb_img']  # images.shape[0]
    for model_param in param['model_list']:
        print(f"Model: {model_param['name']}")
        model = init_model(param, device, model_param['name'], model_param['model'])
        images = np.load(param['org_file'] + model_param['name'] + '.npy')
        adv_img = np.load(param['adv_file'] + model_param['name'] + '.npy')
        img_dist = []
        adv_dist = []

        # Same class
        for i in range(nb_img-1):
            print(f'Image: {i+1}')
            data = torch.from_numpy(images[i]).reshape((1, 1, 28, 28)).to(device)
            output = model(data)
            pred_i = output.argmax(dim=1, keepdim=True).detach().numpy()[0, 0]
            for j in range(i+1, nb_img):
                data = torch.from_numpy(images[j]).reshape((1, 1, 28, 28)).to(device)
                output = model(data)
                pred_j = output.argmax(dim=1, keepdim=True).detach().numpy()[0, 0]
                if pred_i == pred_j:
                    src = images[i]
                    dest = images[j]
                    path, loss_list, pred_list = horizontal_path_light(param, src, dest, model, device)
                    img_dist.append(loss_list[-1])

        # All original pairs & advrsarial
        '''
        for i in range(nb_img-1):
            print(f'Image: {i}')
            src = images[i]
            dest = adv_img[i]
            path, loss_list, pred_list = horizontal_path_light(param, src, dest, model, device)
            adv_dist.append(loss_list[-1])
            for j in range(i+1, nb_img):
                src = images[i]
                dest = images[j]
                path, loss_list, pred_list = horizontal_path_light(param, src, dest, model, device)
                img_dist.append(loss_list[-1])
        
        for i in range(nb_img, images.shape[0]):
            print(f'Image: {i}')
            src = images[i]
            dest = adv_img[i]
            path, loss_list, pred_list = horizontal_path_light(param, src, dest, model, device)
            adv_dist.append(loss_list[-1])
        '''
        img_dist = np.array(img_dist)
        adv_dist = np.array(adv_dist)
        np.save(param['img_dist']+model_param['name'], img_dist)
        np.save(param['adv_dist']+model_param['name'], adv_dist)
        figs.append(plot_hist(img_dist, f"Test examples: {model_param['name']}", "Distance", "Number"))
        figs.append(plot_hist(adv_dist, f"Adv examples: {model_param['name']}", "Distance", "Number"))
    return figs


def vis_hist(param):
    figs = []
    for model_param in param['model_list']:
        img_dist = np.load(param['img_dist']+model_param['name']+'.npy')
        adv_dist = np.load(param['adv_dist']+model_param['name']+'.npy')
        figs.append(plot_hist(img_dist, f"Test examples: {model_param['name']}", "Distance", "Number", 30))
        figs.append(plot_hist(adv_dist, f"Adv examples: {model_param['name']}", "Distance", "Number", 10))
        break
    return figs


# ---------------------------------------------------- Main ------------------------------------------------------------


def main():
    param = load_yaml()
    torch.manual_seed(param['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, model = initialize(param, device)
    _ = test_horizontal_light(param, model, device, param['idx'])
    # _ = test_leaves(param, device)
    # _ = vis_hist(param)
    plt.show()


if __name__ == '__main__':
    main()
