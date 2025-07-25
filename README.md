# StabCF
StabCF: A Stabilized Training Method in Collaborative Filtering

This is our PyTorch implementation for the paper.


## Environment Requirement

The code has been tested running under Python 3.8.0 and torch 2.0.0.

The required packages are as follows:

- pytorch == 2.0.0
- numpy == 1.22.4
- scipy == 1.10.1
- sklearn == 1.1.3
- prettytable == 2.1.0

## Training

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). Important argument:

- `alpha`
  - It controls the relative importance of the current positive instance when mixing with user historical interactions and negative samples.
- `N`
  - It specifies the number of historical positives used for generating the positive samples.
- `M`
  - It specifies the size of negative candidate set when using StabCF.

#### ApeGNN_StabCF
```
python main.py --dataset ali --gnn ApeGNN_HT --lr 0.001 --l2 0.001 --pool sum --ns stabcf --n_negs 16 --alpha 24 --window_length 10 

python main.py --dataset yelp2018 --gnn ApeGNN_HT --lr 0.001 --l2 0.001 --pool sum --ns stabcf --n_negs 16 --alpha 23 --window_length 10 

python main.py --dataset amazon --gnn ApeGNN_HT --lr 0.001 --l2 0.001 --pool sum --ns stabcf --n_negs 16 --alpha 58 --window_length 10 
```
#### LightGCN_StabCF

```
python main.py --dataset ali --lr 0.001 --l2 0.001 --ns stabcf --alpha 21 --n_negs 32 --window_length 10

python main.py --dataset yelp2018 --lr 0.001 --l2 0.001 --ns stabcf --alpha 20 --n_negs 64 --window_length 5

python main.py --dataset amazon --lr 0.001 --l2 0.001 --ns stabcf --alpha 21 --n_negs 64 --window_length 5
```

#### NGCF_StabCF
```
python main.py --dataset ali --gnn ngcf --batch_size 1024 --lr 0.0001 --l2 0.0001 --pool concat --ns stabcf --n_negs 32 --alpha 1 --window_length 5 

python main.py --dataset yelp2018 --gnn ngcf -batch_size 1024 --lr 0.0001 --l2 0.0001 --pool concat --ns stabcf --n_negs 64 --alpha 9 --window_length 5 

python main.py --dataset amazon --gnn ngcf -batch_size 1024 --lr 0.0001 --l2 0.0001 --pool concat --ns stabcf --n_negs 32 --alpha 30 --window_length 5 
```






