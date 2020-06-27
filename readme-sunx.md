## tree generation

### Exp1 feature induced tree

1. 使用预训练模型为每个类别提取特征，用于产生诱导树（代替权重）。
```
python exp1/get_class_features.py --dataset=CIFAR100 --arch=wrn28_10_cifar100 --hierarchy=induced-wrn28_10_cifar100 --pretrained --loss=SoftTreeSupLoss --analysis=HardEmbeddedDecisionRules --tree-supervision-weight=1
```

2. 利用特征产生诱导树，注意，method参数设为induced2
```
python nbdt/bin/nbdt-hierarchy --dataset=CIFAR100 --arch=wrn28_10_cifar100 --method=induced2
```

CIFAR10 feature tree和weight tree看起来差不多；
CIFAR100 feature tree 看起来很差

### Exp2 fusion tree
todo