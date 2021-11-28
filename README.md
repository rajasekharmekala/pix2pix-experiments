# pix2pix-experiments
Implementation of pix2pix from scratch and experiments on various settings on the research paper


#### Train
```
<!-- Training (Note that there are no scripts for downloading datasets mentioned in the paper. The dataset paths and model options can be configured in configuration.py) -->

python train.py --dataset maps --n_layers 3


```

#### Test/Val

```
<!-- Running on validation/test set -->
python evaluate.py 
```

Inspired from Aladdin Persson gan tutorials