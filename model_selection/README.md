# AutoTFB

## Introduction


We provide a automated time series forecasting framework which via both model selection and model ensemble.

Here is the model selection pipeline. The implementation code for model ensemble is located in the [ensemble](./TFB/ts_benchmark/baselines/ensemble/ensemble.py) directory.

The below figure provides a visual overview of AutoTFB's automated time series forecasting framework.
<div align="center">
<img alt="Logo" src="docs/figures/Pipeline.png" width="80%"/>
</div>




## Quickstart

1. Installation:

- From PyPI

Given a python environment (**note**: this project is fully tested under **python 3.8**), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```


2. Data preparation:

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link). Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model:

We provide the experiment scripts for all benchmarks under the folder `./scripts/multivariate_forecast`, and `./scripts/univariate_forecast`. For example you can reproduce a experiment result as the following:

```shell
sh ./scripts/multivariate_forecast/ILI_script/DLinear.sh
```

## Steps to develop your own method
We provide tutorial about how to develop your own method, you can [click here](../docs/tutorials/steps_to_develop_your_own_method.md).


## Steps to evaluate on your own time series
We provide tutorial about how to evaluate on your own time series, you can [click here](../docs/tutorials/steps_to_evaluate_your_own_time_series.md).

## Time series code bug the drop-last illustration
Implementations of existing methods often  employ a “Drop Last” trick in the testing phase. To accelerate the testing, it is common to split the data into batches. However, if we discard the last incomplete batch with fewer instances than the batch size, this causes unfair comparisons. For example, in Figure 4, the ETTh2 has a testing series of length 2,880, and we need to predict 336 future time steps using a look-back window of size 512. If we select the batch sizes to be 32, 64, and 128, the number of samples in the last batch are 17, 49, and 113, respectively. **Unless all methods use the same batch size, discarding the last batch of test samples is unfair, as the actual usage length of the test set is inconsistent.** Table 2 shows the testing results of PatchTST, DLinear, and FEDformer on the ETTh2 with different batch sizes and the “Drop Last” trick turned on. **We observe that the performance of the methods changes when varying the batch size.**

**Therefore, AutoTFB calls for the testing process to avoid using the drop-last operation to ensure fairness, and AutoTFB did not use the drop-last operation during testing either.**
<div align="center">
<img alt="Logo" src="docs/figures/Drop-last.png" width="70%"/>
</div>
