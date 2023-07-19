import sys

sys.path.append("/cluster/scratch/afelderer/code")

import argparse

from ZeDiAc.definitions import LABELCONFIG
from ZeDiAc.Predictors import ZSPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", "-t", choices=["loss", "unexpected"], type=str, default="loss")
    parser.add_argument("--iteration", "-i", type=int, default=3)
    parser.add_argument("--dataset_name", "-d", type=str, default="evaluate")
    args = parser.parse_args()

    templates = LABELCONFIG["templates"]
    labels = LABELCONFIG["labels"][args.target_label]
    hypotheses = {t.format(l): f"D_{templates[t]}_{labels[l]}" for t in templates for l in labels}

    zs = ZSPredictor(
        model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        dataset_name="active-learning",
        hypotheses=hypotheses,
        target_class="unexpected",
        ensemble_temperature=2)

    zs.predict()

if __name__ == "__main__":
    main()
