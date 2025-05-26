# Continuous Evolution Pool: Taming Recurring Concept Drift in Online Time Series Forecasting

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
