# Training & Evaluating  DeepValve 

This page explains how to train and evaluate the DeepValve models (regression and segmentation).

## 2. Preparing the data

This repository is based on a pre-defined input data structure. Our dataloader object, for example, requires coordinate annotations (for regression task ground truths) or a set of 2D masks (for segmentation task ground truths).

Given annotations are provided, and correct configurations such as mask width and number of points are correctly set at `segmentation utils` in the data directory, one can create the masks by running

```sh
python3 data/segmentation_data/create_data.py
```
**Disclaimer:** Code execution relies on default settings as specified in each model's main scripts or the segmentation model's [configuration file](/code/segmentation/config.py). Additionally, per the [main readme](../README.md), prior data addition to your directories is required.

## 3. Training the regression model


 - running the U-net regression model.

    ```sh
    python3 /code/regression/unet_main.py
    ```

 - running the DSNT regression model

    ```sh
    python3 /code/regression/dsnt_main.py
    ```


## 4. Training the segmentation model


 - running the U-net segmentation model

    ```sh
    python3 /code/segmentation/main.py
    ```

## 5. Evaluations

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