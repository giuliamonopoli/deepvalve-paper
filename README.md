# Supplementary code for the paper: DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging


This repository contains supplementary code for the paper
> Giulia Monopoli et al.
> DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging.\
> Submitted to *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 2024.


## Abstract
Early detection of mitral valve (MV) structural complications is a crucial clinical need. Key advances in deep learning-based
segmentation have not yet been leveraged for MV detection in cardiac
magnetic resonance imaging (CMR), a promising opportunity towards
assisted MV disease diagnostics. To address this gap, we introduce Deep-
Valve, an automated pipeline for MV detection using CMR. DeepValve
builds on existing approaches by comparing known methodologies (U-
Net architectures, fine-tuned for regression and segmentation analysis:
UNET-REG and UNET-SEG) and introducing a novel hybrid model
(DSNT-REG), adapted from a recent automatic segmentation study in
echocardiography. We propose metrics tailored for quality assessment of
predicted thin structures based on Procrustes analysis.

## Our pipeline and examples of predictions from our models


The base structure of the pipeline can be seen below
<p align="center">
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/main/figs/PIPE.png"/>
<p align="center">


An example of predictions of the regression model can be seen in the following figure:
<p align="center">
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/main/figs/regresSion_plot4.png"/>
<p align="center">

Finally, an example of the segmentation model's prediction can be seen below:
<p align="center">
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/main/figs/SEG_COMPARISON3.png"/>
<p align="center">

## Getting started
**Note:** Date privacy restricts sharing of the original dataset and annotations. The code in this repository can be adapted to your own purposes. It is not intended to be able to perform automatic mitral valve detection in an end-to-end fashion in its current state.

1. Clone the Repository
  ```sh
git clone https://github.com/giuliamonopoli/deepvalve-paper.git
 ```


### Pre-processing

This repository is based on a pre-defined input data structure. Our dataloader object, for example, requires coordinate annotations (for regression task ground truths) or a set of 2D masks (for segmentation task ground truths).

### Running simulation

For clear instructions on how to run the models, we refer to the [code directory readme](/code/README.md).


### Postprocessing
Add steps for postprocessing / reproducing figures and tables in the paper, ...


## Having issues
If you have any troubles please file and issue in the GitHub repository.
