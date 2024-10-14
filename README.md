# Deep Learning-Based Private Information Inference for Alternating-Offer Bargaining 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Constrained Optimization-Based Valuation Inference

### Training with GRU  Network

To train the BLInfer-C models with synthetic and real bargaining datasets, run the two commands:
```train
python train_BLInfer_C_on_syndata.py --fold_lambda <syndata type> --epsilon 0.001 --k 3 --split_idx 1 --save_root <save root path>
```

```train
python train_BLInfer_C_on_realdata.py --cluster_path <clustering result path> --k 3 --iter_num 7 --epsilon 0.001 --split_idx 1 --save_root <save root path>
```

To train the BLInfer models with synthetic and real bargaining datasets, run the two commands:

```train
python train_BLInfer_on_syndata.py --fold_lambda <syndata type> --epsilon 0.001 --split_idx 1 --save_root <save root path>
```

```train
python train_BLInfer_on_realdata.py --epsilon 0.001 --split_idx 1 --save_root <save root path>
```


### Training with Transformer Network

To train the BLInfer-C models with synthetic and real bargaining datasets, run the two commands:

```train
python Transformer_train_BLInfer_C_on_syndata.py --fold_lambda <syndata type> --epsilon 0.001 --k 3 --split_idx 1 --save_root <save root path>
```

```train
python Transformer_train_BLInfer_C_on_realdata.py --cluster_path <clustering result path> --k 3 --iter_num 7 --epsilon 0.001 --split_idx 1 --save_root <save root path>
```

## Notes
```Note-1
"-- file_root" in these python files need to be aligned with the root where you put the dataset
```
