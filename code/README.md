### Outline of the code structure

#### For regression-based and DSNT-based models

 - testing the U-net regression model.
    
    ```sh
    python3 \code\regression\unet_main.py
    ```

 - testing the DSNT regression model
    
    ```sh
    python3 \code\regression\dsnt_main.py
    ```

#### For segmentation-based models

 - testing the U-net model
    disclaimer: will be run with the default configurations detailed in the [segmentation configuration](\code\segmentation\config.py) but also some other arbitrary parameters set at the [main.py](\code\segmentation\main.py)
    ```sh
    python3 \code\segmentation\main.py
    ```

