# Masked Autoencoders Are Scalable Vision Learners with PyTorch in Juypter Notebook

A PyTorch implementation of Masked Autoencoders Are Scalable Vision Learners [1] by IcarusWizard at <a href="https://github.com/IcarusWizard/MAE">MAE</a> converted into Jupyter notebook format with some quality-of-life utility features such as an early stopper and ETA tracking. The version of Pytorch has also been updated from `1.10.1` to `2.3.0`.

The model has been tested on the CIFAR10 dataset due to limitations in compute resources; models have been trained on a 3060Ti graphics card.

<div align="center">
  <img src=assets/mae.png/><br>
  <small>Source: <a href=https://arxiv.org/abs/2111.06377>Masked Autoencoders Are Scalable Vision Learners</a></small>
</div><br>

## Installation

`pip install -r requirements.txt`

## Usage

Run the cells inside of notebooks `mae_pretrain.ipynb` and `mae_classifier.ipynb`.

See logs using `tensorboard --logdir logs`.

## Results

### Loss

With a patience of 30 epochs, early stopping triggered on MAE training at 1,883 epochs. However, loss is still trending towards decrease. Results could be improved by increasing the patience of the early stopper.

<div align="center">
  <img src=assets/mae_loss.JPG/><br>
  <small>MAE training loss</small>
</div><br>

<div align="center">
  <img src=assets/mae_classification_loss.JPG/><br>
  <small>Classification train and validation loss</small>
</div><br>

### Accuracy

| Model              | Validation Accuracy |
| ------------------ | ------------------- |
| ViT-T w/o pretrain | 71.8                |
| ViT-T w/ pretrain  | **85.64**           |

There is significantly less overfitting in the model results with MAE pretraining

<div align="center">
  <img src=assets/mae_classification_accuracy.JPG/><br>
  <small>Train vs. validation accuracy</small>
</div><br>

Weights are available in [GitHub Release](https://github.com/teddyld/mae-torch/releases/tag/v1.0.0)

## Future work

I am also doing my own research into the application of MAE in engagement detection on the much larger DAiSEE dataset so I will be releasing weights on that in future releases

- [ ] Release DAiSEE weights

## References

[1] Masked Autoencoders Are Scalable Vision Learners; He et al.; arXiv 2021; https://arxiv.org/abs/2111.06377.
