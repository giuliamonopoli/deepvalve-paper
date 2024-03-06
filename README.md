# Supplementary code for the paper: DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging


This repository contains supplementary code for the paper
> Giulia Monopoli et al.
> DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging


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
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/adding-code-and-deps/figs/DVpipeline.png"/>
<p align="center">

pred_regression

Some predictions of the regression model can be seen at
<p align="center">
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/adding-code-and-deps/figs/regresSion_plot4.pdf"/>
<p align="center">

While some of the segmentations can be seen below:
<p align="center">
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/adding-code-and-deps/figs/segm_model.png"/>
<p align="center">

## Getting started
1. Clone the Repository
  ```sh
git clone https://github.com/giuliamonopoli/deepvalve-paper.git
 ```


### Pre-processing

This repository assumes some previous structure. Our dataloader object, for example, requires some sort of coordinate annotations for regression tasks and also a previous set of masks to be used as a ground truth for the segmentation models.

### Running simulation

For clear instructions on how to run the models, we refer to the [code directory readme](/code/README.md).


### Postprocessing
Add steps for postprocessing / reproducing figures and tables in the paper, ...


## Having issues
If you have any troubles please file and issue in the GitHub repository.
