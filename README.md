# SurvivalAnalysis

This repository contains simulation code and multiple survival analysis models implemented in PyTorch and Scikit-learn. It supports both linear and nonlinear simulation setups for evaluating survival prediction models.

## Implemented Models

- **Cox-nnet**: A shallow neural network trained with the Cox proportional hazards loss.
- **DeepSurv**: A deep neural network variant of the Cox model.
- **RSF (Random Survival Forest)**: A tree-based ensemble model using the `scikit-survival` library.
- **Linear Regression (for comparison)**: A basic regression baseline on log survival times.

## Simulation Settings

- **Simu20**: Linear hazard function based simulation with continuous and binary covariates.
- **Simu33**: Complex simulation including squared terms and interaction effects, modeling nonlinear structure.

## Evaluation Metrics

All models are evaluated using:
- **Concordance Index (C-index)**: Measures ranking accuracy of predicted risk scores.
- **Mean Absolute Error (MAE)**: Estimates deviation between true and predicted event times.
- *(IBS and other metrics may be added in the future.)*

## Project Structure

- `simu20coxnnet.py`: Cox-nnet on Simu20
- `simu20deepsurv.py`: DeepSurv on Simu20
- `simu20RfSurv.py`: Random Survival Forest on Simu20
- `simulations20.py`: Simulation utilities for Simu20

- `simu33coxnnet.py`: Cox-nnet on Simu33
- `simu33deepsurv.py`: DeepSurv on Simu33
- `simu33RfSurv.py`: Random Survival Forest on Simu33
- `simulations33.py`: Simulation utilities for Simu33

- `rsf_sim_result.csv`, `cindex_result_nn20_seed_1.csv`, etc.: Output evaluation results.

## Future Work

If time allows, we may incorporate:
- **DeepHit**
- **Transformer-based survival models**
- **Calibration metrics**, **IBS**, and more

