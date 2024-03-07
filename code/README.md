After successful installation of the [requirements](../requirements.txt),


#### For regression-based and DSNT-based models

 - running the U-net regression model.
    
    ```sh
    python3 /code/regression/unet_main.py
    ```

 - running the DSNT regression model
    
    ```sh
    python3 /code/regression/dsnt_main.py
    ```
    

#### For segmentation-based models

 - running the U-net segmentation model

    ```sh
    python3 /code/segmentation/main.py
    ```

**Disclaimer:** Code execution relies on default settings as specified in each model's main scripts or the segmentation model's [configuration file](/code/segmentation/config.py). Additionally, per the [main readme](../README.md), prior data addition to your directories is required.
