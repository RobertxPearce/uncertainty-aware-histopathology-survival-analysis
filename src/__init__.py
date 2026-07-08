# Public API for the histopathology survival-analysis package.
#
#   load_survival_table  ->  make_datasets  ->  make_dataloaders     (data)
#   build_model / build_optimizer                                    (model)
#   train                                                            (training)
#   predict / evaluate                                               (prediction & evaluation)
#   mc_dropout_predict / deep_ensemble_predict / sngp_predict        (uncertainty)

from .data import (
    make_survival_metadata,
    attach_feature_paths,
    make_splits,
    load_survival_table,
    make_datasets,
    make_dataloaders,
    make_dataloaders_from_csv,
    SurvivalBagDataset,
    collate_bags,
)
from .models import (
    ABMILSurvival,
    build_model,
    SNGPABMILSurvival,
    build_sngp_model,
)
from .losses import (
    cox_loss,
    survrnc_loss,
    survrnc_cox_loss,
)
from .train import (
    build_optimizer,
    train_one_epoch,
    evaluate_epoch,
    train,
)
from .eval import (
    concordance_index,
    predict,
    evaluate,
    bootstrap_cindex,
    mc_dropout_predict,
    deep_ensemble_predict,
    fit_sngp_covariance,
    sngp_predict,
)
from .utils import (
    seed_everything,
    worker_init_fn,
    make_generator,
    pick_device,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # data loading -> datasets -> dataloaders
    "make_survival_metadata",
    "attach_feature_paths",
    "make_splits",
    "load_survival_table",
    "make_datasets",
    "make_dataloaders",
    "make_dataloaders_from_csv",
    "SurvivalBagDataset",
    "collate_bags",

    # model construction
    "ABMILSurvival",
    "build_model",
    "SNGPABMILSurvival",
    "build_sngp_model",
    "build_optimizer",

    # loss
    "cox_loss",
    "survrnc_loss",
    "survrnc_cox_loss",

    # training
    "train",
    "train_one_epoch",
    "evaluate_epoch",

    # prediction and evaluation
    "predict",
    "evaluate",
    "concordance_index",
    "bootstrap_cindex",

    # uncertainty estimation
    "mc_dropout_predict",
    "deep_ensemble_predict",
    "fit_sngp_covariance",
    "sngp_predict",

    # utilities
    "seed_everything",
    "worker_init_fn",
    "make_generator",
    "pick_device",
    "save_checkpoint",
    "load_checkpoint",
]
