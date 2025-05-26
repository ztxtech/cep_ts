<div align="center">
  <h2><b> Continuous Evolution Pool: Taming Recurring Concept Drift
    <br/> in Online Time Series Forecasting </b></h2>
</div>

**Repo Status:**

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
[![Visits Badge](https://badges.pufler.dev/visits/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts)
[![GitHub last commit](https://img.shields.io/github/last-commit/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts/activity?ref=master&activity_type=direct_push)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts/graphs/commit-activity)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts)
[![GitHub Repo stars](https://img.shields.io/github/stars/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts)
[![GitHub forks](https://img.shields.io/github/forks/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts)
[![GitHub watchers](https://img.shields.io/github/watchers/ztxtech/cep_ts)](https://github.com/ztxtech/cep_ts)

## Introduction

Accurate time series forecasting is a fundamental task in various fields such as finance, energy management, traffic prediction, and environmental monitoring. However, online time series forecasting often faces a significant challenge known as concept drift, especially recurring concept drift. Recurring concept drift refers to the periodic reappearance of certain data patterns after a period of absence. Existing solutions mainly rely on parameter - updating techniques, which may lead to the loss of previously learned knowledge and lack effective knowledge retention mechanisms.

## The Continuous Evolution Pool (CEP)

To address these limitations, we propose the Continuous Evolution Pool (CEP), a novel pooling mechanism designed to store multiple forecasters corresponding to different concepts. When a new test sample arrives, CEP selects the nearest forecaster in the pool for prediction and learns from the features of its neighboring samples. If there are insufficient neighboring samples, it indicates the emergence of a new concept, and a new model will be added to the pool. Additionally, CEP employs an elimination mechanism to remove outdated knowledge and filter noisy data.

## Main Contributions

1. We identify recurring concept drift as a significant challenge in online time series forecasting and point out the shortcomings of existing methods.
2. We propose the CEP mechanism, which effectively mitigates knowledge loss and enables the model to leverage previously acquired knowledge more effectively.
3. Through extensive experiments on multiple real - world datasets and various neural network architectures, we demonstrate that CEP substantially enhances prediction accuracy in scenarios characterized by recurring concept drift.

## Installation

1. Clone the repository:
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Extract the Dataset

```bash
unzip ./data/data.zip -d ./data/
```

### Manually Running Experiments

1. Run the main script:

```bash
python main.py --some_parameter XXX
```

### Running Experiments with Configurations

1. Run the main script with a configuration file:

```bash
python main.py --config ./configs/main/ep_cep_da_ECL_ml_CEP_fr_Crossformer_pn_1_fo_0.8_do_1.5_oe_fade_pt_True.json
```

### Reproducing Experiments from the Paper

1. Main Experiment: Run the script in `./run_scripts`:

```bash
python run_scripts/cep.py
```

2. Ablation Study: To verify CEP components effectiveness:

```bash
python run_scripts/ablation.py
```

3. Baseline Comparison: Reproduce baseline methods results:

```bash
python run_scripts/baseline.py
```

4. Parameter Sensitivity: Test key parameters' impacts:

```bash
python run_scripts/parameter_sensitivity.py
```

## Repository Structure

- `configs`: Contains configuration files for different experiments.
- `data`: Includes data loading scripts and datasets.
- `exp`: Contains experiment scripts.
- `layers`: Contains neural network layer implementations.
- `models`: Contains different model implementations.
- `utils`: Contains utility functions.

## Citation

If you find this repo useful, please cite our paper.

```
@misc{zhan2025continuous,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Tianxiang Zhan and Ming Jin and Yuanpeng He and Yuxuan Liang and Yong Deng and Shirui Pan},
  year={2025}
}
```

## License

This source code is released under the MIT license, included [here](LICENSE).

## Acknowledgement

This library is constructed based on the following repos:

- [**FSNet** https://github.com/salesforce/fsnet/](https://github.com/salesforce/fsnet/).
- [**OneNet**: https://github.com/yfzhang114/OneNet/](https://github.com/yfzhang114/OneNet/).
- [**Time Series Library**: https://github.com/thuml/Time-Series-Library/](https://github.com/thuml/Time-Series-Library/).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ztxtech/cep_ts&type=Date)](https://star-history.com/#ztxtech/cep_ts&Date)
