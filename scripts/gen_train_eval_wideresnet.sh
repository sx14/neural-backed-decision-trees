# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.

MODEL_NAME="wrn28_10"
# for i in "CIFAR10 ${MODEL_NAME}_cifar10 1" "CIFAR100 ${MODEL_NAME}_cifar100 1" "TinyImagenet200 ${MODEL_NAME} 10"; do
for i in "CIFAR100 ${MODEL_NAME}_cifar100 1"; do
  read dataset model weight <<< "${i}";

  # 1. generate hieararchy
  nbdt-hierarchy  --dataset=${dataset} --arch=${model} --method=induced

  # 2. train with soft tree supervision loss
  python main.py --lr=0.01 --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --pretrained --loss=SoftTreeSupLoss --tree-supervision-weight=${weight}
  # --lr=0.01 --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --pretrained --loss=SoftTreeSupLoss --tree-supervision-weight=1

  # 3. evaluate with soft then hard inference
  for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
    python main.py --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --eval --resume --analysis=${analysis} --tree-supervision-weight=${weight}
    # --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --loss=SoftTreeSupLoss --eval --resume --analysis=SoftEmbeddedDecisionRules --tree-supervision-weight=1
  done
done;
