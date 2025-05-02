# Supplementary code for the paper: DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging


This repository contains supplementary code for the paper
> Giulia Monopoli et al.
> DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging.\
> Published to *Computers in biology and medicine*, June 2024.


## Abstract
Mitral valve (MV) assessment is key to diagnosing valvular disease and to addressing serious downstream complications.
Cardiac magnetic resonance (CMR) has become an essential diagnostic tool in MV disease, offering detailed views of the valve structure and function, while also overcoming the limitations of other imaging modalities. Automated detection of the MV leaflets in CMR  could enable rapid and precise assessments that enhance diagnostic accuracy,  yet this has not been accomplished.

To address this gap, we introduce DeepValve, the first  deep learning  (DL) pipeline for MV detection using CMR. Within DeepValve, we tested three valve detection models: a keypoint-regression model (UNET-REG), a segmentation model (UNET-SEG) and a hybrid model based on keypoint detection (DSNT-REG). We also propose  metrics, newly tailored, for evaluating the quality of MV detection, including Procrustes-based metrics  for the regression models and customized Dice-based metrics for segmentation models. We developed and tested our models on a clinical dataset comprising 120 CMR images from patients with confirmed MV disease (mitral valve prolapse and mitral annular disjunction). 

Our results show that DSNT-REG delivered the best regression performance as reflected in traditional and newly-applied metrics, accurately approximating landmarks locations and precisely determining their spatial orientation. 
UNET-SEG achieved satisfactory Dice and customized Dice scores, reflecting the segmentation's model's ability to accurately predict valve location and topology.
Overall, our work represents a critical first step towards automated MV assessment using DL in CMR and paving the way for improved clinical assessment in MV disease.

## Our pipeline and examples of predictions from our models


The base structure of the project's pipeline can be seen below
<p align="center">
<img src="https://github.com/giuliamonopoli/deepvalve-paper/blob/main/figs/DeepValve_pipeline.png"/>
<p align="center">


## Getting started
**Note:** Date privacy restricts sharing of the original dataset and annotations. The code in this repository can be adapted to your own purposes. It is not intended to be able to perform automatic mitral valve detection in an end-to-end fashion in its current state.

1. Clone the Repository
  ```sh
git clone https://github.com/giuliamonopoli/deepvalve-paper.git
 ```

2. Install the [requirements](requirements.txt).

### Pre-processing

This repository is based on a pre-defined input data structure. Our dataloader object, for example, requires coordinate annotations (for regression task ground truths) or a set of 2D masks (for segmentation task ground truths).

Given annotations are provided, and correct configurations such as mask width and number of points are correctly set at [segmentation utils](/data/segmentation_data/utils.py) in the data directory, one can create the masks by running 

```sh
python3 data/segmentation_data/create_data.py
```

### Training

For clear instructions on how to run the models, we refer to the [code directory readme](/code/README.md).


### Evaluations

For the evaluation of the model's predictions, one should run 

```sh
python3 code/evaluate_models.py
```

That will generate .txt files with the appropriate evaluation metrics of the trained models to the unseen data.
and plot those predictions.
Customised dice scores can be obtained running
```sh
python3 code/dice_customised.py
```
### Uncertainty quantification in models' predcitions
```sh
python3 code/uq_dsnt.py or code/uq_seg.py
```

## Having issues
If you have any troubles please file and issue in the GitHub repository.
