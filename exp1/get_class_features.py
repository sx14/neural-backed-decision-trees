"""
Neural-Backed Decision Trees training script on CIFAR10, CIFAR100, TinyImagenet200

The original version of this `main.py` was taken from

    https://github.com/kuangliu/pytorch-cifar

and extended in

    https://github.com/alvinwan/pytorch-cifar-plus

The script has since been heavily modified to support a number of different
configurations and options. See the current repository for a full description
of its bells and whistles.

    https://github.com/alvinwan/neural-backed-decision-trees
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nbdt import data, analysis, loss, models

import torchvision
import torchvision.transforms as transforms

import os
import pickle
import argparse
import numpy as np

from nbdt.utils import (
    progress_bar, generate_fname, generate_kwargs, Colors, maybe_install_wordnet
)

maybe_install_wordnet()

datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=100, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--arch', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# extra general options for main script
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--pretrained', action='store_true',
                    help='Download pretrained model. Not all models support this.')
parser.add_argument('--eval', help='eval only', action='store_true')

# options specific to this project and its dataloaders
parser.add_argument('--loss', choices=loss.names, default='CrossEntropyLoss')
parser.add_argument('--analysis', choices=analysis.names, help='Run analysis after each epoch')
parser.add_argument('--input-size', type=int,
                    help='Set transform train and val. Samples are resized to '
                    'input-size + 32.')
parser.add_argument('--lr-decay-every', type=int, default=0)

data.custom.add_arguments(parser)
loss.add_arguments(parser)
analysis.add_arguments(parser)

args = parser.parse_args()

loss.set_default_values(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = getattr(data, args.dataset)

if args.dataset in ('TinyImagenet200', 'Imagenet1000'):
    default_input_size = 64 if args.dataset == 'TinyImagenet200' else 224
    input_size = args.input_size or default_input_size
    transform_train = dataset.transform_train(input_size)
    transform_test = dataset.transform_val(input_size)
elif args.input_size is not None and args.input_size > 32:
    transform_train = transforms.Compose([
        transforms.Resize(args.input_size + 32),
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(args.input_size + 32),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

dataset_kwargs = generate_kwargs(args, dataset,
    name=f'Dataset {args.dataset}',
    keys=data.custom.keys,
    globals=globals())

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

Colors.cyan(f'Training with dataset {args.dataset} and {len(trainset.classes)} classes')

# Model
print('==> Building model..')
model = getattr(models, args.arch)
model_kwargs = {'num_classes': len(trainset.classes) }

if args.pretrained:
    print('==> Loading pretrained model..')
    try:
        net = model(pretrained=True, dataset=args.dataset, **model_kwargs)
    except TypeError as e:  # likely because `dataset` not allowed arg
        print(e)
        
        try:
            net = model(pretrained=True, **model_kwargs)
        except Exception as e:
            Colors.red(f'Fatal error: {e}')
            exit()
else:
    net = model(**model_kwargs)

    checkpoint_fname = generate_fname(**vars(args))
    checkpoint_path = './checkpoint/{}.pth'.format(checkpoint_fname)
    print(f'==> Checkpoints will be saved to: {checkpoint_path}')

    # TODO(alvin): fix checkpoint structure so that this isn't needed
    def load_state_dict(state_dict):
        try:
            net.load_state_dict(state_dict)
        except RuntimeError as e:
            if 'Missing key(s) in state_dict:' in str(e):
                net.load_state_dict({
                    key.replace('module.', '', 1): value
                    for key, value in state_dict.items()})


    resume_path = args.path_resume or checkpoint_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if not os.path.exists(resume_path):
        print('==> No checkpoint found. Skipping...')
    else:
        checkpoint = torch.load(resume_path, map_location=torch.device(device))

        if 'net' in checkpoint:
            load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            Colors.cyan(f'==> Checkpoint found for epoch {start_epoch} with accuracy '
                        f'{best_acc} at {resume_path}')
        else:
            load_state_dict(checkpoint)
            Colors.cyan(f'==> Checkpoint found at {resume_path}')

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True



criterion = nn.CrossEntropyLoss()
class_criterion = getattr(loss, args.loss)
loss_kwargs = generate_kwargs(args, class_criterion,
    name=f'Loss {args.loss}',
    keys=loss.keys,
    globals=globals())
criterion = class_criterion(**loss_kwargs)


def get_word2vecs(classes):
    obj2vec_path = os.path.join('nbdt', 'hierarchies', args.dataset, 'class_vectors.bin')
    if os.path.exists(obj2vec_path):
        with open(obj2vec_path, 'rb') as f:
            obj2vecs = pickle.load(f)
        return obj2vecs

    import gensim
    obj2vecs = np.zeros((len(classes), 300))
    w2v_path = os.path.join('exp1', 'GoogleNews-vectors-negative300.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print('loading pretrained word2vec done.')
    # vrd object labels to vectors
    for i in range(len(classes)):
        obj_label = classes[i]
        print('[%d] %s' % (i, obj_label))
        vec = model[obj_label]
        if vec is None or len(vec) == 0 or np.sum(vec) == 0:
            print('[WARNING] %s' % obj_label)
            exit(-1)
        obj2vecs[i] = vec

    with open(obj2vec_path, 'wb') as f:
        pickle.dump(obj2vecs, f)

    return obj2vecs

def get_class_features(epoch, analyzer):
    save_path = os.path.join('nbdt', 'hierarchies', args.dataset, 'class_features.bin')
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            class_features = pickle.load(f)
        return class_features

    analyzer.start_test(epoch)

    feature_len = 640
    if 'wrn28_10' not in args.arch:
        print('[sunx] Unsupported arch for feature extraction.')
        exit(0)

    class_features = np.zeros((len(trainset.classes), feature_len))
    class_counter = np.zeros(len(trainset.classes))

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = net.features(inputs)
            features = features.view(features.size(0), -1)
            outputs = net.output(features)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            hits = predicted.eq(targets)
            correct += hits.sum().item()

            if device == 'cuda':
                outputs = outputs.cpu()
                targets = targets.cpu()
                features = features.cpu()
                hits = hits.cpu()

            features = features.numpy()
            for i in range(targets.shape[0]):
                if hits[i].item() == 1:
                    cid = targets[i].item()
                    fea = features[i]
                    class_features[cid] = class_features[cid] + fea
                    class_counter[cid] = class_counter[cid] + 1

            stat = analyzer.update_batch(outputs, targets)
            extra = f'| {stat}' if stat else ''

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) %s'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, extra))

    analyzer.end_test(epoch)

    # print(class_counter.sum())
    for i in range(class_counter.shape[0]):
        class_features[i] = class_features[i] / class_counter[i]

    with open(save_path, 'wb') as f:
        pickle.dump(class_features, f)
    print('Class features are saved at %s' % save_path)
    return class_features


class_analysis = getattr(analysis, args.analysis or 'Noop')
analyzer_kwargs = generate_kwargs(args, class_analysis,
    name=f'Analyzer {args.analysis}',
    keys=analysis.keys,
    globals=globals())
analyzer = class_analysis(**analyzer_kwargs)


if not args.pretrained:
    Colors.red(' * Warning: Model is loaded from checkpoint. ')
else:
    Colors.red(' * Warning: Model is loaded with pre-trained weights. ')

analyzer.start_epoch(0)
class_w2v = get_word2vecs(trainset.classes)
class_fea = get_class_features(0, analyzer)
exit()
