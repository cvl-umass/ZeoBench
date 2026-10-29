"""Bootstrap held-out test metrics for Zeolite and MOF downstream runs."""

import argparse
import glob
import json
import os
import traceback

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

from solo.data.classification_dataloader import build_grid_dataset
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel


def fast_bootstrap_mse(predictions, targets, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if not len(predictions) or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same nonzero length.")
    squared_errors = (predictions - targets) ** 2
    indices = rng.integers(0, len(predictions), size=(n_bootstraps, len(predictions)))
    bootstrapped_mses = np.mean(squared_errors[indices], axis=1)
    return (
        float(np.mean(bootstrapped_mses)),
        float(np.percentile(bootstrapped_mses, 2.5)),
        float(np.percentile(bootstrapped_mses, 97.5)),
        float(np.std(bootstrapped_mses)),
        float(np.median(bootstrapped_mses)),
    )


def fast_bootstrap_ap(predictions, targets, n_bootstraps=1000, seed=42):
    """Bootstrap Average Precision for the Zeolite synthesizability task."""
    rng = np.random.default_rng(seed)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets).reshape(-1)
    bootstrapped_aps = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(targets), size=len(targets))
        y_true_boot = targets[indices]
        if len(np.unique(y_true_boot)) > 1:
            bootstrapped_aps.append(average_precision_score(y_true_boot, predictions[indices]))
    if not bootstrapped_aps:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return (
        float(np.mean(bootstrapped_aps)),
        float(np.percentile(bootstrapped_aps, 2.5)),
        float(np.percentile(bootstrapped_aps, 97.5)),
        float(np.std(bootstrapped_aps)),
        float(np.median(bootstrapped_aps)),
    )


def has_valid_bootstrap_outputs(metrics_path, csv_eval_path):
    """Avoid skipping a run when a previous evaluator left partial files."""
    if not os.path.isfile(metrics_path) or not os.path.isfile(csv_eval_path):
        return False
    try:
        with open(metrics_path, "r") as stream:
            saved_metrics = json.load(stream)
        required_metrics = ("Bootstrapped_MSE_Mean", "Bootstrapped_MSE_Std")
        if any(not np.isfinite(float(saved_metrics[key])) for key in required_metrics):
            return False
        saved_evaluations = pd.read_csv(csv_eval_path)
        required_columns = {"Split", "Prediction", "Target"}
        has_sample_id = "Sample_ID" in saved_evaluations.columns or "Zeolite" in saved_evaluations.columns
        return (
            required_columns.issubset(saved_evaluations.columns)
            and has_sample_id
            and saved_evaluations["Split"].astype(str).str.lower().eq("test").any()
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _plain_mapping(value):
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _valid_label(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_text(row, key):
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _prediction_items(logits):
    """Keep class score vectors intact while flattening scalar regression output."""
    values = logits.detach().cpu().numpy()
    if values.ndim == 0:
        return [float(values)]
    if values.ndim == 1:
        return values.tolist()
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0].tolist()
    return [value for value in values]


def _dataset_and_target(row, cfg):
    dataset = (_row_text(row, "Dataset") or str(cfg.data.get("dataset", "zeolites"))).lower()
    target = _row_text(row, "Target")
    if dataset == "zeolites":
        molecule = (_row_text(row, "Molecule") or target).replace("zeolites_", "")
        if not molecule:
            molecule = str(_plain_mapping(cfg.data.get("dataset_kwargs", {})).get("target", "C18"))
        if molecule.upper() == "SYNTHESIZABILITY":
            molecule = "synthesizability"
        else:
            molecule = molecule.upper()
        target = molecule
    elif dataset == "mofs":
        if not target:
            kwargs = _plain_mapping(cfg.data.get("dataset_kwargs", {}))
            target = kwargs.get("annotations_file", "MOF adsorption").removesuffix(".csv")
    else:
        raise ValueError(f"Unsupported bootstrap dataset: {dataset}")
    return dataset, target


def _dataset_cache_key(dataset, root, target, kwargs, use_handcrafted):
    return json.dumps(
        [dataset, os.path.realpath(root), target, kwargs, bool(use_handcrafted)],
        sort_keys=True,
        default=str,
    )


def _build_split_datasets(dataset, root, target, cfg, use_handcrafted):
    kwargs = _plain_mapping(cfg.data.get("dataset_kwargs", {}))
    if dataset == "zeolites":
        kwargs.pop("target", None)
        test_dataset = build_grid_dataset(
            dataset, root, is_train=False, is_val=False, transform=None,
            dataset_kwargs={"target": target, "full": False, "use_handcrafted": use_handcrafted},
        )
        val_dataset = build_grid_dataset(
            dataset, root, is_train=False, is_val=True, transform=None,
            dataset_kwargs={"target": target, "full": False, "use_handcrafted": use_handcrafted},
        )
    else:
        test_dataset = build_grid_dataset(
            dataset, root, is_train=False, is_val=False, transform=None,
            dataset_kwargs=kwargs,
        )
        val_dataset = build_grid_dataset(
            dataset, root, is_train=False, is_val=True, transform=None,
            dataset_kwargs=kwargs,
        )
    return test_dataset, val_dataset


def _evaluate_cached_dataset(dataset, model, cached_features):
    sample_id_key = getattr(dataset, "sample_id_key", "zeolite")
    target_key = getattr(dataset, "target_key", "kH_C18")
    names, features, targets = [], [], []
    for item in dataset.files:
        sample_id = item[sample_id_key]
        label = item.get(target_key, float("nan"))
        if _valid_label(label) and sample_id in cached_features:
            names.append(sample_id)
            features.append(cached_features[sample_id])
            targets.append(float(label))
    if not features:
        return [], np.array([]), np.array([])
    feature_tensor = torch.stack(features).to(dtype=torch.float32)
    with torch.no_grad():
        predictions = _prediction_items(model(feature_tensor)["logits"])
    return names, np.asarray(predictions), np.asarray(targets)


def _evaluate_dataset(dataset, model, cfg, device, num_workers):
    sample_id_key = getattr(dataset, "sample_id_key", "zeolite")
    target_key = getattr(dataset, "target_key", "kH_C18")
    names = [item[sample_id_key] for item in dataset.files if _valid_label(item.get(target_key))]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            valid_indices = torch.isfinite(targets)
            if not valid_indices.any():
                continue
            inputs = inputs[valid_indices].to(device)
            targets = targets[valid_indices].detach().cpu().numpy().reshape(-1)
            if not getattr(cfg.performance, "disable_channel_last", False):
                if inputs.ndim == 5:
                    inputs = inputs.to(memory_format=torch.channels_last_3d)
                elif inputs.ndim == 4:
                    inputs = inputs.to(memory_format=torch.channels_last)
            logits = model(inputs)["logits"]
            all_predictions.extend(_prediction_items(logits))
            all_targets.extend(targets.tolist())
    return names, np.asarray(all_predictions), np.asarray(all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the task dispatch CSV")
    parser.add_argument("--data_dir", type=str, default="", help="Fallback dataset root for legacy task CSVs")
    parser.add_argument("--task_id", type=int, default=0, help="Slurm Array Task ID")
    parser.add_argument("--num_tasks", type=int, default=1, help="Total number of Slurm Array Tasks")
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader worker count")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.num_tasks <= 0 or args.task_id < 0 or args.task_id >= args.num_tasks:
        raise SystemExit("--task_id must be within the positive --num_tasks range.")
    chunk_indices = np.array_split(df.index, args.num_tasks)[args.task_id]
    chunk_df = df.loc[chunk_indices]
    print(f"Worker {args.task_id}/{args.num_tasks} processing {len(chunk_df)} models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cache = {}

    for _, row in chunk_df.iterrows():
        exact_dir = str(row["Exact_Directory"])
        print(f"\n[{args.task_id}] Processing: {exact_dir}")
        try:
            metrics_path = os.path.join(exact_dir, "test_bootstrap_metrics.json")
            csv_eval_path = os.path.join(exact_dir, "raw_evaluations.csv")
            if has_valid_bootstrap_outputs(metrics_path, csv_eval_path):
                print(f"[{args.task_id}] -> Skipping, valid metrics and raw evaluations already exist.")
                continue

            ckpt_files = glob.glob(os.path.join(exact_dir, "*.ckpt"))
            args_path = os.path.join(exact_dir, "args.json")
            if not ckpt_files or not os.path.exists(args_path):
                print(f"[{args.task_id}] -> Missing checkpoint or args.json. Skipping.")
                continue
            ckpt_path = ckpt_files[0]
            with open(args_path, "r") as stream:
                cfg = OmegaConf.create(json.load(stream))

            dataset_name, target_name = _dataset_and_target(row, cfg)
            saved_data_root = cfg.data.get("val_path")
            data_root = _row_text(row, "Data_Directory") or args.data_dir or str(saved_data_root or "")
            if not data_root:
                raise ValueError("No dataset root was provided in the dispatch row, CLI, or saved config.")
            use_handcrafted = bool(cfg.data.get("dataset_kwargs", {}).get("use_handcrafted", False))
            kwargs_for_key = _plain_mapping(cfg.data.get("dataset_kwargs", {}))
            cache_key = _dataset_cache_key(dataset_name, data_root, target_name, kwargs_for_key, use_handcrafted)
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = _build_split_datasets(
                    dataset_name, data_root, target_name, cfg, use_handcrafted
                )
            test_dataset, val_dataset = dataset_cache[cache_key]

            if "_target_" in cfg.backbone:
                backbone = hydra.utils.instantiate(cfg.backbone)
            else:
                backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
                backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
            if hasattr(backbone, "fc"):
                backbone.fc = torch.nn.Identity()
            if hasattr(backbone, "heads"):
                backbone.heads.head = torch.nn.Identity()

            saved_checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = saved_checkpoint.get("state_dict", {})
            if "classifier.weight" in state_dict:
                backbone.num_features = state_dict["classifier.weight"].shape[1]
            model = LinearModel.load_from_checkpoint(ckpt_path, backbone=backbone, cfg=cfg)
            model = model.to(device).eval()

            feature_cache_path = None
            pretrained_feature_extractor = cfg.get("pretrained_feature_extractor")
            if cfg.get("cache_features", False) and pretrained_feature_extractor:
                possible_cache = os.path.join(os.path.dirname(pretrained_feature_extractor), "offline_features.pt")
                if os.path.exists(possible_cache):
                    feature_cache_path = possible_cache

            if feature_cache_path:
                print(f"[{args.task_id}] -> Evaluating via offline cache.")
                model.caching_features = True
                cached_features = torch.load(feature_cache_path, map_location=device, weights_only=False)
                test_names, test_predictions, y_test = _evaluate_cached_dataset(test_dataset, model, cached_features)
                val_names, val_predictions, y_val = _evaluate_cached_dataset(val_dataset, model, cached_features)
                if not len(test_names):
                    print(f"[{args.task_id}] -> Cache key mismatch. Skipping.")
                    continue
            else:
                print(f"[{args.task_id}] -> Evaluating from saved validation transform.")
                model.caching_features = False
                transform = hydra.utils.instantiate(cfg.val_transform)
                test_dataset.transform = transform
                val_dataset.transform = transform
                test_names, test_predictions, y_test = _evaluate_dataset(
                    test_dataset, model, cfg, device, args.num_workers
                )
                val_names, val_predictions, y_val = _evaluate_dataset(
                    val_dataset, model, cfg, device, args.num_workers
                )
                test_dataset.transform = None
                val_dataset.transform = None

            if not len(test_names):
                raise ValueError(f"No labeled samples found in the {dataset_name} test split for {target_name}.")
            if dataset_name == "zeolites" and target_name.lower() == "synthesizability":
                ap_scores = test_predictions[:, 1] if test_predictions.ndim == 2 else test_predictions
                mean_val, ci_lower, ci_upper, std_val, median_val = fast_bootstrap_ap(ap_scores, y_test)
            else:
                mean_val, ci_lower, ci_upper, std_val, median_val = fast_bootstrap_mse(test_predictions, y_test)

            metrics_dict = {
                "Bootstrapped_MSE_Mean": mean_val,
                "Bootstrapped_MSE_95CI_Lower": ci_lower,
                "Bootstrapped_MSE_95CI_Upper": ci_upper,
                "Bootstrapped_MSE_Std": std_val,
                "Bootstrapped_MSE_Median": median_val,
                "Dataset": dataset_name,
                "Task_Target": target_name,
                "Test_Sample_Count": int(len(y_test)),
            }
            with open(metrics_path, "w") as stream:
                json.dump(metrics_dict, stream, indent=4, allow_nan=True)

            sample_id_key = getattr(test_dataset, "sample_id_key", "zeolite")
            eval_data = []
            for split, names, predictions, targets in (
                ("val", val_names, val_predictions, y_val),
                ("test", test_names, test_predictions, y_test),
            ):
                for name, prediction, target_value in zip(names, predictions, targets):
                    record = {
                        "Sample_ID": name,
                        "Dataset": dataset_name,
                        "Task_Target": target_name,
                        "Split": split,
                        "Prediction": prediction,
                        "Target": target_value,
                    }
                    if sample_id_key == "zeolite":
                        record["Zeolite"] = name
                    eval_data.append(record)
            pd.DataFrame(eval_data).to_csv(csv_eval_path, index=False)
            print(f"[{args.task_id}] -> Saved metrics and raw evaluations successfully.")
        except Exception as exc:
            print(f"[{args.task_id}] -> ERROR evaluating {exact_dir}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
