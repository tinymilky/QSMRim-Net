# QSMRim-Net
 
# [QSMRim-Net: Imbalance-Aware Learning for Identification of Chronic Active Multiple Sclerosis Lesions on Quantitative Susceptibility Maps](https://www.biorxiv.org/content/10.1101/2022.01.31.478482v1?rss=1) 
![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg) <br>

[Hang Zhang](https://tinymilky.github.io/), Thanh D. Nguyen, Jinwei Zhang, Melanie Marcille, Pascal Spincemaille, Yi Wang, Susan A. Gauthier, [Elizabeth M. Sweeney](https://emsweene.github.io/). (bioRxiv preprint, submitted to NeuroImage: Clinical)

## Background

Chronic active multiple sclerosis (MS) lesions are characterized by a paramagnetic rim at the edge of the lesion and are associated with increased disability in patients. Quantitative susceptibility mapping (QSM) is an MRI technique that is sensitive to chronic active lesions, termed rim+ lesions on the QSM. We present QSMRim-Net, a data imbalance-aware deep neural network that fuses lesion-level radiomic and convolutional image features for automated identification of rim+ lesions on QSM. 

## Usage

The dataset used to verify the performance of the proposed method is unvailable per the policy of [Weill Cornell Medicine](https://weill.cornell.edu/). 
However, algorithms mentioned in the [QSMRim-Net paper](https://www.biorxiv.org/content/10.1101/2022.01.31.478482v1?rss=1) are available in this repositorty.

We use a simple U-Net as backbone to show how our RSA block can be pugged into existing network. <br>
`./src/QSMRim-Net.py` contains QSMRim-Net architecture with detailed modules import from `./src/backbones/resnet.py`. The configurations for extracting radiomic features using [PyRadiomics](https://pyradiomics.readthedocs.io/) are listed in `radiomics/config.yaml`.

## QSMRim-Net Framwork

<div align=center><img width=90% src="/figs/network_architecture.png"/></div>

Schematic of the proposed QSMRim-Net for paramagnetic rim lesion identification. (Top) The deep residual network takes in both QSM and FLAIR images to extract convolutional features. (Bottom) The QSM image and the lesion mask are used to extract radiomic features, followed by feature extraction of an MLP. A tensor concatenation operation is performed to fuse convolutional and radiomic features, and a DeepSMOTE layer is used to perform synthetic minority feature over-sampling during the training phase.

## Deep Synthetic Minority Oversampling TEchnique (DeepSMOTE)

<div align=center><img width=65% src="/figs/smote_layer.png"/></div>

Schematic of the DeepSMOTE network layer. N is the number of samples in a training mini-batch, and n is the number of rim+ samples in the mini-batch. The input features go through an MLP for feature transformation, followed by selecting rim+ samples out from the mini-batch. Then the transformed rim+ features are used to generate the similarity using Euclidean distance followed by latent feature interpolation and concatenation of the original feature and the oversampled feature, resulting in total N+2n samples in the output of DeepSMOTE.

## Rim+ Lesion Example

<div align=center><img width=75% src="/figs/rim_example.png"/></div>

Example of MS lesions on an axial slice of the QSM (left) and corresponding axial slice of the T2-FLAIR (right). The digit 1 marked with red indicates a rim+ lesion and the digit 2 marked with green indicates a rim- lesion. (More details in the [paper](https://www.biorxiv.org/content/10.1101/2022.01.31.478482v1?rss=1))

## Citation
If you are inspired by [QSMRim-Net](https://www.biorxiv.org/content/10.1101/2022.01.31.478482v1?rss=1) or use our [code](https://github.com/tinymilky/QSMRim-Net), please cite:
```
@article {Zhang2022.01.31.478482,
	author = {Zhang, Hang and Nguyen, Thanh D. and Zhang, Jinwei and Marcille, Melanie and Spincemaille, Pascal and Wang, Yi and Gauthier, Susan A. and Sweeney, Elizabeth M.},
	title = {QSMRim-Net: Imbalance-Aware Learning for Identification of Chronic Active Multiple Sclerosis Lesions on Quantitative Susceptibility Maps},
	elocation-id = {2022.01.31.478482},
	year = {2022},
	doi = {10.1101/2022.01.31.478482},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/02/01/2022.01.31.478482},
	eprint = {https://www.biorxiv.org/content/early/2022/02/01/2022.01.31.478482.full.pdf},
	journal = {bioRxiv}
}

```