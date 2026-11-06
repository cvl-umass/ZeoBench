# ZeoBench: A Benchmark for Self-Supervised Learning on 3D Zeolite Representations
[site](https://cvl-umass.github.io/ZeoBench/) arxiv [data](https://github.com/cvl-umass/ZeoBench/blob/main/docs/ZEOLITE_DATA_PREPROCESSING.md)

Built on the [solo-learn](https://github.com/vturrisi/solo-learn) framework and codebase.
The environment information is provided in `requirements.txt` and the python version used was python 3.10.20.
## Preparing the Zeolite Dataset

This repository uses distance grids stored as `distance-grids-h5/{IZASC,PCOD}/<ID>.h5py`. The download and preprocessing workflow, including Zeo++ setup and generation at 0.15 Å grid resolution, is described in [the Zeolite data preprocessing guide](docs/ZEOLITE_DATA_PREPROCESSING.md). The guide also explains how to compare a small CPU Slurm sample with an existing processed dataset. It supports selected structures and deterministic shards, so the full dataset need not be regenerated for a small run.

## Pretraining

Self-supervised pretraining is handled by `main_pretrain.py`. 

All pretraining configurations used in our experiments (e.g., DINO, BYOL, MoCo, SimCLR) are located in the `configs/pretrain/` directory.
Before running, you must point the configurations to your local copy of the Zeolites dataset. You can do this by either editing the `train_path` and `val_path` inside the respective `.yaml` config file, or by overriding them directly via the command line.
The supplied pretraining configs default to two GPU devices. Adjust `devices`, batch size, and related settings in the config to match your hardware.

### Running a Pretraining Experiment

To launch a pretraining run (for example, using DINO), use the following command:

```
python main_pretrain.py \
    --config-path configs/pretrain \
    --config-name dino_pretrain.yaml \
    data.train_path=/absolute/path/to/dataset/zeolites \
    data.val_path=/absolute/path/to/dataset/zeolites
```

## Feature Extraction (Baselines & SSL Models)

To significantly speed up linear probing and hyperparameter sweeping, we precompute and cache the backbone features into an `offline_features.pt` file. This is handled by `precompute_features.py`.
The script supports both extracting features from custom SSL checkpoints and extracting features directly from ImageNet baselines.

### 1. Extracting Features for SSL Models (MoCo, BYOL, DINO, etc.)

For models trained via `main_pretrain.py`, you must provide the path to the resulting `.ckpt` file.

```
python precompute_features.py \
    --config-path configs/pretrain \
    --config-name dino_pretrain.yaml \
    data.train_path=/absolute/path/to/dataset/zeolites \
    data.val_path=/absolute/path/to/dataset/zeolites \
    pretrained_feature_extractor=/absolute/path/to/trained_models/dino/model.ckpt
```

### 2. Extracting Features for Baselines (ImageNet, 3DINO)

For our ImageNet-adapted baselines and 3DINO, the configuration files inherently bypass standard checkpoint loading and fetch the appropriate pretrained weights automatically.
Note for 3DINO: You must first download the official [3DINO-ViT](https://github.com/AICONSlab/3DINO) weights and specify the path in configs/pretrain/3dino_pretrain.yaml under backbone.pretrained_weights.
To run extraction for a baseline (e.g., Inflated ResNet), simply run:
```
python precompute_features.py \
    --config-path configs/pretrain \
    --config-name inflated_resnet_D60_scale_sqrt_res0.15_pretrain.yaml \
    data.train_path=/absolute/path/to/dataset/zeolites \
    data.val_path=/absolute/path/to/dataset/zeolites
```
If desired, the test time augmentation count can be varied with `num_tta_views` parameter.

## Linear Probing

For linear probing, we freeze the backbone and train only a linear head. To make this process incredibly efficient (especially for hyperparameter sweeps), we rely on the `offline_features.pt` file generated in the **Feature Extraction** step. 

When `cache_features=true` is set, the data loaders will bypass standard image processing and directly load the precomputed features, cutting down training time drastically.

Linear-probe configurations are organized by feature extractor and dataset under `configs/linear_probe/<feature-family>/`. For example, BYOL C18 10k uses `configs/linear_probe/byol/zeolites_c18_sub-10000.yaml`. For SSL features, first create the feature cache with `precompute_features.py`, then point `pretrained_feature_extractor` to the resulting `offline_features.pt`.

### Running a Linear Probe

To run a linear probe using your precomputed features (for example, on the C18 dataset using DINO features):

```bash
python main_linear.py \
    --config-path configs/linear_probe/byol \
    --config-name zeolites_c18_sub-10000.yaml \
    data.train_path=/absolute/path/to/dataset/zeolites \
    data.val_path=/absolute/path/to/dataset/zeolites \
    data.dataset_kwargs.custom_train_file=/absolute/path/to/subsets/train_subset_C18_10000.txt \
    pretrained_feature_extractor=/absolute/path/to/offline_features.pt
```
Key Parameters:
- pretrained_feature_extractor: Must point to the directory containing offline_features.pt or the file itself.
- data.dataset_kwargs.target: Set the target molecule (e.g., C18, C30).
- data.subset_size: Adjust the subset size (e.g., 10000, 1000, 100).

## Fine-Tuning
For fine-tuning, we train both the backbone and the linear head end-to-end. This requires passing the actual PyTorch Lightning checkpoint (.ckpt) rather than the cached features, and setting finetune=true.
Dataset-specific fine-tuning configurations are under `configs/fine_tune/<family>/`, including separate directories for SSL methods and scratch baselines.

### Running Fine-Tuning
To fine-tune an existing self-supervised model (e.g., BYOL) on a downstream task:
```
python main_linear.py \
    --config-path configs/fine_tune/byol \
    --config-name zeolites_c18_sub-10000.yaml \
    data.train_path=/absolute/path/to/dataset/zeolites \
    data.val_path=/absolute/path/to/dataset/zeolites \
    data.dataset_kwargs.custom_train_file=/absolute/path/to/subsets/train_subset_C18_10000.txt \
    pretrained_feature_extractor=/absolute/path/to/trained_models/byol/model.ckpt
```
### Training From Scratch (Supervised Baseline)
To train a supervised model from scratch (e.g., ResNet18 or ViT), use the provided from_scratch configurations and omit the pretrained_feature_extractor argument:
```
python main_linear.py \
    --config-path configs/fine_tune/scratch_resnet \
    --config-name zeolites_c18_sub-10000.yaml \
    data.train_path=/absolute/path/to/dataset/zeolites \
    data.val_path=/absolute/path/to/dataset/zeolites \
    data.dataset_kwargs.custom_train_file=/absolute/path/to/subsets/train_subset_C18_10000.txt
```

## MOF Fine-Tuning

Full-data MOF regression configs are provided under `configs/fine_tune/<family>/mofs_*_full.yaml` for CO2 and CH4 adsorption at 0.05, 0.5, and 2.5 bar. They use the `adsorption` value from the matching `hmof_<GAS>_<PRESSURE>bar.csv`. Training uses `train_set.txt`; validation and test remain held out. Point `data.train_path` and `data.val_path` to the local MOF dataset root.

```bash
python main_linear.py \
    --config-path configs/fine_tune/byol \
    --config-name mofs_ch4_0p05bar_full.yaml \
    data.train_path=/path/to/ML-MOFs \
    data.val_path=/path/to/ML-MOFs \
    pretrained_feature_extractor=/path/to/byol_checkpoint.ckpt
```

Use the matching family directory and config for another method. For a from-scratch baseline, use a `scratch_*` config and omit the checkpoint override. MOF grids are loaded from `<dataset-root>/distance-grids-h5/hMOF/<MOF-ID>.h5py`.

## Invariance Evaluation

To evaluate how invariant our learned representations are to spatial transformations, we measure the standard deviation of the model's predictions across 100 deterministic rotations and translations. This is handled by `main_invariance.py`.
As reported in our study, we primarily run invariance evaluations on the C18 molecule using the 10,000 subset. 

### 1. Evaluating Fine-Tuned or Trained-from-Scratch Models

For models trained end-to-end (e.g., a ResNet trained from scratch or a fine-tuned SSL model), point the script to the Hydra output directory of the linear/fine-tuning run. The script will automatically load the full weights from `last.ckpt`.
```
# Evaluate rotational invariance
python main_invariance.py \
    --job_dir /absolute/path/to/hydra_out/linear/scratch_resnet/ \
    --mode rotate \
    --num_augs 100
```
Note: You can isolate the evaluation by changing --mode to rotate or translate.
### 2. Evaluating Linear Probed Models (e.g., 3DINO)

For models evaluated via linear probing on cached offline_features.pt (like 3DINO or frozen ImageNet baselines), the saved last.ckpt only contains the linear classifier weights. main_invariance.py automatically detects this and will load the base backbone configuration natively while only applying the linear weights.

Important for 3DINO: Because 3DINO uses a custom normalization and a specific crop size that is deeply nested in the original config, you must pass these overrides explicitly via the command line:
```
python main_invariance.py \
    --job_dir /absolute/path/to/hydra_out/linear/3dino_linear/ \
    --mode rotate \
    --use_3dino_norm \
    --crop_size 112
```
### Outputs
The script writes `invariance_results_<mode>.csv` in the provided `--job_dir` (for example, `invariance_results_rotate.csv` or `invariance_results_both.csv`). Each file contains the prediction-variance measurements for the test set.

## Bootstrapping Error Bars

The evaluator supports Zeolite and MOF runs. MOF predictions are read from the target-specific annotations and split files; bootstrap intervals use the held-out test split.
To generate the standard deviations used in our plots, we perform bootstrapping on the test set predictions. This is handled by `evaluate_bootstraps.py`.
Because evaluating hundreds of models can be time-consuming, this script is designed to process models in parallel. It reads a CSV file containing the directories of the completed models and splits the workload based on a provided "Task ID".

### 1. Preparing the Evaluation List

First, compile a CSV of the models you wish to evaluate. It must contain:
- `Exact_Directory`: The absolute path to the model's Hydra output directory (containing `args.json` and a `.ckpt`).

The evaluator reads the dataset and target from each saved config by default. Optional columns override those values:
- `Dataset`: `zeolites` or `mofs`.
- `Molecule`: The Zeolite target (for example, `C18`).
- `Target`: An explicit target name, such as a MOF adsorption target.
- `Data_Directory`: The dataset root for this model. Use this for mixed Zeolite/MOF lists when their roots differ.

### 2. Running the Evaluation

To evaluate a chunk of models, specify the total number of tasks (`--num_tasks`) and the current worker's ID (`--task_id`). 
For example, to run the script locally as a single process (1 task, ID 0):

```
python evaluate_bootstraps.py \
    --csv_path /path/to/your/compiled_models.csv \
    --data_dir /absolute/path/to/dataset/zeolites \
    --num_tasks 1 \
    --task_id 0
```

### Outputs
For each model processed, the script will generate two new files within the model's original Exact_Directory:
- raw_evaluations.csv: A complete log of the model's predictions vs. targets for every zeolite in the validation and test sets.
- test_bootstrap_metrics.json: A summary file containing the Bootstrapped Mean, Median, Standard Deviation, and the 95% CI Lower/Upper bounds (using 1,000 resamples).

These JSON files can then be easily crawled by a visualization notebook to plot the final scaling curves.
