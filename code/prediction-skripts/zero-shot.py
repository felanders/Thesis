import sys

sys.path.append("/cluster/home/afelderer/Thesis/code")

import argparse

from ZeDiAc.definitions import LABELCONFIG, MODELCONFIG
from ZeDiAc.Predictors import ZSPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", "-t", choices=["loss", "unexpected", "unexpected_loss"], type=str, default="loss")
    parser.add_argument("--dataset_name", "-d", type=str, default="evaluate")
    parser.add_argument("--model_name", "-m", type=str, default="DeBERTa")
    parser.add_argument("--ensemble_temperature", type=int, default=2)
    args = parser.parse_args()

    templates = LABELCONFIG["templates"]
    labels = LABELCONFIG["labels"][args.target_class]
    model_name=MODELCONFIG[args.model_name]["name"]
    model_abbrev=MODELCONFIG[args.model_name]["abbrev"]
    hypotheses = {t.format(l): f"{model_abbrev}_{templates[t]}_{labels[l]}" for t in templates for l in labels}

    zs = ZSPredictor(
        model_name=model_name,
        dataset_name=args.dataset_name,
        hypotheses=hypotheses,
        target_class=args.target_class,
        ensemble_temperature=args.ensemble_temperature)

    zs.predict()

if __name__ == "__main__":
    main()
