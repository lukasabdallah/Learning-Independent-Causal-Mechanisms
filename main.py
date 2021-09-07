import argparse
import importlib
import json
import os
import sys
import time
import math
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets as datasets_torch
from torchvision import transforms
import torchvision

from model import Expert, Discriminator
from utils import init_weights
import matplotlib.pyplot as plt
from copy import copy


def initialize_expert(epochs, expert, i, optimizer, loss, data_train, args, writer):
    print("Initializing expert [{}] as identity on preturbed data".format(i+1))
    expert.train()
    loss_list = []
    for epoch in range(epochs):
        total_loss = 0
        n_samples = 0
        total_step = len(data_train)
        # for batch in data_train:
        for step, (x_canonical, x_transf) in enumerate(data_train):
            # x_canonical, x_transf = batch
            batch_size = x_canonical.size(0)
            n_samples += batch_size
            # x_transf = x_transf.view(x_transf.size(0), -1).to(args.device)
            x_hat = expert(x_canonical)

            loss_rec = loss(x_hat, x_canonical)
            loss_list.append(loss_rec)
            total_loss += loss_rec.item()*batch_size
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()
            if (step+1) % 10 == 0:
                print(
                    f'Epoch [{epoch +1}/{epochs}], Step [{step+1}/{total_step}], Loss: {loss_rec}')
                writer.add_scalar(
                    f"Expert_{i}_initialization_loss_per_step", loss_rec, global_step=step+1)

                with torch.no_grad():
                    fake = expert(x_canonical)
                    data = x_canonical
                    img_grid_fake = torchvision.utils.make_grid(fake)
                    img_grid_canonical = torchvision.utils.make_grid(data)

                    writer.add_image(
                        f"Expert{i} Generated MNIST Images", img_grid_fake, global_step=step+1
                    )
                    writer.add_image(
                        f"Expert{i} Corresponding canonical Images", img_grid_canonical, global_step=step+1
                    )
            if ((step+1) == 500 or loss_rec < 0.002):
                break
        else:
            continue
        break
        # Loss
        #mean_loss = total_loss/n_samples
        # print("initialization epoch [{}] expert [{}] loss {:.4f}".format(
        #    epoch+1, i+1, mean_loss))
        # writer.add_scalar('expert_{}_initialization_loss_per_episode'.format(
        #    i+1), mean_loss, epoch+1)

        # if mean_loss < 0.002:
        #    break

    #torch.save(expert.state_dict(), f"{checkpt_dir}/{args.name}_E_{i + 1}_init.pt")
    path = os.path.join(checkpt_dir, f"{args.name}_E_{i + 1}_init.pt")
    # path = f'{checkpt_dir}/{args.name}_E_{i + 1}_init.pt'

    torch.save(expert.state_dict(), path)


def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, data_labels, args, writer, transformation_dict):
    discriminator.train()
    for i, expert in enumerate(experts):
        expert.train()
    total_step = 0
    total_step_per_episode = len(data_train)
    # Labels for canonical vs transformed samples
    canonical_label = 1
    transformed_label = 0

    # Keep track of losses
    total_loss_D_canon = 0
    total_loss_D_transformed = 0
    n_samples = 0
    total_loss_expert = [0 for i in range(len(experts))]
    total_samples_expert = [0 for i in range(len(experts))]
    expert_scores_D = [0 for i in range(len(experts))]
    expert_winning_samples_idx = [[] for i in range(len(experts))]

    for epoch in range(args.epochs):
        # Iterate through data
        for step, (batch_train, batch_label) in enumerate(zip(data_train, data_labels)):
            #x_canon = x_transformed
            #X_transf = identity
            #x_canon, x_transf = batch
            #x_canon, _ = batch_train
            #x_transf, _ = batch_label
            # x_transf = torch.randn(x_canon.size()) # TODO temporary since do not have the preturbed data yet
            # batch_size = x_canon.size(0)
            # n_samples += batch_size
            #x_canon = x_canon.view(batch_size, -1).to(args.device)
            #x_transf = x_transf.view(batch_size, -1).to(args.device)
            # ----------------------------------
            total_step = total_step + 1
            x_noise, transformation_idx = batch_train
            x_canon, _ = batch_label
            batch_size = x_noise.size(0)
            n_samples += batch_size
            # print(x_canon.size())

            # Train Discriminator on canonical distribution
            scores_canon = discriminator(x_canon)
            loss_D_canon = criterion(
                scores_canon, torch.ones_like(scores_canon))
            total_loss_D_canon += loss_D_canon.item() * batch_size
            optimizer_D.zero_grad()
            loss_D_canon.backward()

            # Train Discriminator on experts output
            # labels.fill_(transformed_label)
            loss_D_transformed = 0
            exp_outputs = []
            expert_scores = []
            for i, expert in enumerate(experts):
                exp_output = expert(x_noise)
                # print(exp_output.size())
                #exp_outputs.append(exp_output.view(batch_size, 1, args.input_size))
                exp_outputs.append(exp_output)
                exp_scores = discriminator(exp_output.detach())
                expert_scores.append(exp_scores)
                loss_D_transformed += criterion(exp_scores,
                                                torch.zeros_like(exp_scores))

            for sample_idx, transf_idx in enumerate(transformation_idx):
                expert_scores_dict = {}
                # expert_scores_tensor = torch.as_tensor(expert_scores)
                # print(expert_scores_tensor.squeeze().size())
                # print(expert_scores_tensor.squeeze())
                # [tensor[[][][][][]], [[][][][][]], [[][][][][]]]]
                for i in range(len(expert_scores)):
                    expert = i
                    score = expert_scores[i].squeeze()[sample_idx].item()
                    expert_scores_dict[f"expert{expert}"] = score
                writer.add_scalars(
                    f'D(E(X))__{transformation_dict[transf_idx]}', expert_scores_dict, global_step=total_step)

            loss_D_transformed = loss_D_transformed / args.num_experts
            total_loss_D_transformed += loss_D_transformed.item() * batch_size
            loss_D_transformed.backward()
            optimizer_D.step()

            # Train experts
            exp_outputs = torch.cat(exp_outputs, dim=1)
            # print("exp_outputs")
            # print(exp_outputs.size())
            expert_scores = torch.cat(expert_scores, dim=1)
            # print(expert_scores.size())
            mask_winners = expert_scores.argmax(dim=1)
            # Update each expert on samples it won
            for i, expert in enumerate(experts):
                winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
                # print("winning_indexes")
                # print(winning_indexes)
                accrue = 0 if step == 0 else 1
                expert_winning_samples_idx[i] += (winning_indexes +
                                                  accrue*n_samples).tolist()
                n_expert_samples = winning_indexes.size(0)
                if n_expert_samples > 0:
                    total_samples_expert[i] += n_expert_samples
                    exp_samples = exp_outputs[winning_indexes, i].unsqueeze(
                        dim=1)
                    # print("exp_samples")
                    # print(exp_samples.size())
                    D_E_x_transf = discriminator(exp_samples.detach())
                    loss_E = criterion(
                        D_E_x_transf, torch.ones_like(D_E_x_transf))
                    total_loss_expert[i] += loss_E.item() * n_expert_samples
                    optimizers_E[i].zero_grad()
                    # TODO figure out why retain graph is necessary
                    # loss_E.backward(retain_graph=True)
                    loss_E.backward()
                    optimizers_E[i].step()
                    expert_scores_D[i] += D_E_x_transf.squeeze().sum().item()

                    # exp_samples_grid = torchvision.utils.make_grid(exp_samples)
                    # writer.add_image(
                    #     f'Expert_{i}_won_samples', exp_samples_grid, global_step=total_step)
                    # writer.add_scalar(
                    #     f"Expert_{i}_discriminator_loss_for_won_samples", torch.mean(D_E_x_transf), global_step=total_step)
                    # writer.add_scalar(
                    #     f"Expert_{i}_loss_for_won_samples", loss_E, global_step=total_step)

            if (step+1) % 10 == 0:
                print(
                    f'Epoch [{epoch +1}], Step [{step+1}/{total_step_per_episode}], Total Step [{total_step}]')
                writer.add_scalar(
                    f"Discriminator_loss_average_over_all_experts", loss_D_transformed, global_step=total_step)
                writer.add_scalar(
                    f"Discriminator_loss_canonical", loss_D_canon, global_step=total_step)

            if (total_step == 2000):
                break
        else:
            continue
        break
    # Logging
    # mean_loss_D_generated = total_loss_D_transformed / n_samples
    # mean_loss_D_canon = total_loss_D_canon / n_samples
    # print("epoch [{}] loss_D_transformed {:.4f}".format(
    #     epoch + 1, mean_loss_D_generated))
    # print("epoch [{}] loss_D_canon {:.4f}".format(
    #     epoch + 1, mean_loss_D_canon))
    # writer.add_scalar('loss_D_canonical', mean_loss_D_canon, epoch + 1)
    # writer.add_scalar('loss_D_transformed', mean_loss_D_generated, epoch + 1)
    # for i in range(len(experts)):
    #     print("epoch [{}] expert [{}] n_samples {}".format(
    #         epoch + 1, i + 1, total_samples_expert[i]))
    #     writer.add_scalar('expert_{}_n_samples'.format(
    #         i + 1), total_samples_expert[i], epoch + 1)
    #     writer.add_text('expert_{}_winning_samples'.format(i + 1),
    #                     ":".join([str(j) for j in expert_winning_samples_idx[i]]), epoch + 1)
    #     if total_samples_expert[i] > 0:
    #         mean_loss_expert = total_loss_expert[i] / total_samples_expert[i]
    #         mean_expert_scores = expert_scores_D[i] / total_samples_expert[i]
    #         print("epoch [{}] expert [{}] loss {:.4f}".format(
    #             epoch + 1, i + 1, mean_loss_expert))
    #         print("epoch [{}] expert [{}] scores {:.4f}".format(
    #             epoch + 1, i + 1, mean_expert_scores))
    #         writer.add_scalar('expert_{}_loss'.format(
    #             i + 1), mean_loss_expert, epoch + 1)
    #         writer.add_scalar('expert_{}_scores'.format(
    #             i + 1), mean_expert_scores, epoch + 1)

    print("Saving models")
    torch.save(discriminator.state_dict(), checkpt_dir +
               '/{}_D.pt'.format(args.name))
    for i in range(args.num_experts):
        torch.save(experts[i].state_dict(), checkpt_dir +
                   '/{}_E_{}.pt'.format(args.name, i+1))


class AddGaussianNoise(object):
    def __init__(self, mean=0., variance=1.):
        self.variance = variance
        self.mean = mean

    def __call__(self, tensor):
        tensor = tensor + torch.randn(tensor.size()) * \
            math.sqrt(self.variance) + self.mean
        return torch.clamp(tensor, min=0.0, max=1.0)


class ContrastInversion(object):

    def __init__(self):
        pass

    def __call__(self, img):

        return transforms.functional.invert(img)


class Subset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform, label):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.label = label

    def __getitem__(self, idx):
        im, _ = self.dataset[self.indices[idx]]

        return self.transform(im), self.label

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    # Arguments

    # default_settings for Optimizer:
    # α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10−8
    parser = argparse.ArgumentParser(
        description='Learning Independent Causal Mechanisms')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--outdir', default='.', type=str,
                        help='path to the output directory')
    parser.add_argument('--dataset', default='patient', type=str,
                        help='name of the dataset')
    parser.add_argument('--optimizer_experts', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--optimizer_discriminator', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--optimizer_initialize', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--input_size', type=int, default=1024, metavar='N',
                        help='input size of data (default: 1024)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs_init', type=int, default=100, metavar='N',
                        help='number of epochs to initially train experts (default: 10)')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--learning_rate_initialize', type=float, default=1e-1,
                        help='size of expert learning rate')
    parser.add_argument('--learning_rate_expert', type=float, default=1e-3,
                        help='size of expert learning rate')
    # 2e-3 , 5e-3 for discriminator
    parser.add_argument('--learning_rate_discriminator', type=float, default=1e-3,
                        help='size of discriminator learning rate')
    parser.add_argument('--name', type=str, default='',
                        help='name of experiment')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for optimizer')
    parser.add_argument('--num_experts', type=int, default=10, metavar='N',
                        help='number of experts (default: 10)')
    parser.add_argument('--load_initialized_experts', type=bool, default=False,
                        help='whether to load already pre-trained experts')
    parser.add_argument('--model_for_initialized_experts', type=str, default='',
                        help='path to pre-trained experts')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--load_dataset', type=bool, default=False,
                        help='load preprocessed dataset')
    # Get arguments
    args = parser.parse_args()
    # args.device = "cpu"

    print(f"Is cuda available?{torch.cuda.is_available()}")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(args.device)
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Experiment name
    timestamp = str(int(time.time()))
    if args.name == '':
        name = '{}_n_exp_{}_bs_{}_lri_{}_lre_{}_lrd_{}_ei_{}_e_{}_oi_{}_oe_{}_oe_{}_{}'.format(
            args.dataset, args.num_experts, args.batch_size, args.learning_rate_initialize,
            args.learning_rate_expert, args.learning_rate_discriminator, args.epochs_init,
            args.epochs, args.optimizer_initialize, args.optimizer_experts, args.optimizer_discriminator,
            timestamp)
        args.name = name
    else:
        args.name = '{}_{}'.format(args.name, timestamp)
    print('\nExperiment: {}\n'.format(args.name))

    # Logging. To run: tensorboard --logdir <args.outdir>/logs
    log_dir = os.path.join(args.outdir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir_exp = os.path.join(log_dir, args.name)
    os.mkdir(log_dir_exp)
    writer = SummaryWriter(log_dir=log_dir_exp)

    # Directory for checkpoints
    # checkpt_dir = os.path.join(args.outdir, 'checkpoints')
    checkpt_dir = os.path.join(args.outdir, 'checkpoints')

    if not os.path.exists(checkpt_dir):
        os.mkdir(checkpt_dir)

    args.load_dataset = True
    transformation_dict = (
        "left_shift",
        "up_left_shift",
        "up_shift",
        "up_right_shift",
        "right_shift",
        "down_right_shift",
        "down_shift",
        "down_left_shift",
        "gaussian_noise",
        "contrast_inversion"
    )
    if args.load_dataset:

        transf_set = torch.load(os.path.join(args.datadir, 'transf_set.pt'))
        val_set = torch.load(os.path.join(args.datadir, 'val_set.pt'))
    else:
        left_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((0, 2, 4, 2)), ContrastInversion()])
        up_left_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((0, 0, 4, 4)), ContrastInversion()])
        up_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((2, 0, 2, 4)), ContrastInversion()])
        up_right_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((4, 0, 0, 4)), ContrastInversion()])
        right_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((4, 2, 0, 2)), ContrastInversion()])
        down_right_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((4, 4, 0, 0)), ContrastInversion()])
        down_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((2, 4, 2, 0)), ContrastInversion()])
        down_left_shift = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad((0, 4, 4, 0)), ContrastInversion()])
        gaussian_noise = transforms.Compose([transforms.ToTensor(), transforms.Pad(
            2), ContrastInversion(), AddGaussianNoise(0.0, 0.25)])
        contrast_inversion = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad(2)])
        identity = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad(2), ContrastInversion()])

        transformations = (left_shift, up_left_shift, up_shift, up_right_shift, right_shift,
                           down_right_shift, down_shift, down_left_shift, gaussian_noise, contrast_inversion)
        transform = transforms.RandomChoice(transformations)

        dataset_train = datasets_torch.MNIST(
            root=f'{args.datadir}/{args.dataset}', train=True, download=True)
        # MNIST = 60000 = 1875 * 32 (batch_size) pics

        data_set, val_set = torch.utils.data.random_split(
            dataset_train, [30000, 30000])

        data_set.dataset = copy(dataset_train)
        transformed_datasets = []
        for idx, transformation in enumerate(transformations):
            set = Subset(data_set, list(range(
                idx, len(data_set), len(transformations))), transformation, idx)
            transformed_datasets.append(set)

        transf_set = torch.utils.data.ConcatDataset(transformed_datasets)
        val_set.dataset.transform = identity
        # transf_set.dataset.transform = transform
        # val_set.dataset.transform = identity
        torch.save(transf_set, os.path.join(args.datadir, 'transf_set.pt'))
        torch.save(val_set, os.path.join(args.datadir, 'val_set.pt'))
    # Create Dataloader from dataset
    data_train = DataLoader(
        transf_set, batch_size=args.batch_size, shuffle=True
    )
    data_labels = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True
    )
    examples = iter(data_train)
    example_data, example_targets = examples.next()
    more = iter(data_labels)
    more_labels, more_targets = more.next()
    img_grid = torchvision.utils.make_grid(example_data)
    label_grid = torchvision.utils.make_grid(more_labels)
    writer.add_image('mnist_images', img_grid)
    writer.add_image('mnist_labels', label_grid)

    # Model
    experts = [Expert() for i in range(args.num_experts)]
    discriminator = Discriminator()

    # Losses
    loss_initial = torch.nn.MSELoss(reduction='mean')
    # apply L1 loss to identity
    # loss_initial = torch.nn.L1Loss(reduction='mean')
    criterion = torch.nn.BCELoss(reduction='mean')

    # Initialize Experts as approximately Identity on Transformed Data
    # load expert
    args.load_initialized_experts = True
    args.model_for_initialized_experts = "./InitializedExperts/patient_n_exp_10_bs_32_lri_0.1_lre_0.001_lrd_0.001_ei_100_e_100_oi_adam_oe_adam_oe_adam_1630888635"
    for i, expert in enumerate(experts):
        if args.load_initialized_experts:
            # path = os.path.join(checkpt_dir,
            #                     args.model_for_initialized_experts + '_E_{}_init.pt'.format(i+1))
            path = f"{args.model_for_initialized_experts}_E_{i+1}_init.pt"
            init_weights(expert, path)
        else:
            if args.optimizer_initialize == 'adam':
                optimizer_E = torch.optim.Adam(expert.parameters(), lr=args.learning_rate_initialize,
                                               weight_decay=args.weight_decay)
            elif args.optimizer_initialize == 'sgd':
                optimizer_E = torch.optim.SGD(expert.parameters(), lr=args.learning_rate_initialize,
                                              weight_decay=args.weight_decay)
            else:
                raise NotImplementedError
            initialize_expert(args.epochs_init, expert, i,
                              optimizer_E, loss_initial, data_train, args, writer)

    # Optimizers
    optimizers_E = []
    for i in range(args.num_experts):
        if args.optimizer_experts == 'adam':
            optimizer_E = torch.optim.Adam(experts[i].parameters(), lr=args.learning_rate_expert,
                                           weight_decay=args.weight_decay)
        elif args.optimizer_experts == 'sgd':
            optimizer_E = torch.optim.SGD(experts[i].parameters(), lr=args.learning_rate_expert,
                                          weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        optimizers_E.append(optimizer_E)
    if args.optimizer_discriminator == 'adam':
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator,
                                       weight_decay=args.weight_decay)
    elif args.optimizer_discriminator == 'sgd':
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate_discriminator,
                                      weight_decay=args.weight_decay)

    train_system(args.epochs, experts, discriminator, optimizers_E,
                 optimizer_D, criterion, data_train, data_labels, args, writer, transformation_dict)
    # # Training
    # for epoch in range(args.epochs):
    #     train_system(epoch, experts, discriminator, optimizers_E,
    #                  optimizer_D, criterion, data_train, data_labels, args, writer)

    #     if epoch % args.log_interval == 0 or epoch == args.epochs-1:
    #         torch.save(discriminator.state_dict(), checkpt_dir +
    #                    '/{}_D.pth'.format(args.name))
    #         for i in range(args.num_experts):
    #             torch.save(experts[i].state_dict(), checkpt_dir +
    #                        '/{}_E_{}.pth'.format(args.name, i+1))
