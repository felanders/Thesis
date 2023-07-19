import sys

sys.path.append("/cluster/home/afelderer/Thesis/code")

import argparse

import datasets
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
from ZeDiAc.definitions import BASE_PATH, TRAININGCONFIG
from ZeDiAc.Trainers import FineTuningTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", "-t", choices=["loss", "unexpected"], type=str, default="loss")
    parser.add_argument("--iteration", "-i", type=int, default=3)
    parser.add_argument("--downsampling_factor", type=int, default=3)
    args = parser.parse_args()

    target_class = args.target_class
    iteration = args.iteration
    downsampling_factor = args.downsampling_factor

    df = pd.read_pickle(BASE_PATH/f"data/processed/active-learning-iteration-{iteration}.pkl")
    df = df[df.labeled == True]
    data = datasets.Dataset.from_pandas(df, preserve_index=False)
    data = data.map(lambda x: {"labels": torch.tensor([x[target_class], 1-x[target_class]], dtype=torch.float32)})
    d_t = data.filter(lambda ex: ex["labels"][0] >= 0.5)
    d_f = data.filter(lambda ex: ex["labels"][0] < 0.5)
    n_t = d_t.shape[0]
    n_f = d_f.shape[0]
    factor = n_f/n_t
    if factor > downsampling_factor:
        data = datasets.concatenate_datasets([d_t, d_f.train_test_split(test_size=d_t.shape[0]*downsampling_factor)["test"]], axis = 0)
        data = data.shuffle(seed=19950808)
    model_name = str(BASE_PATH/f"model/{target_class}/deberta-v3-large-distill")

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenization(inputs):
        result = tokenizer(
            inputs["text"],
            return_tensors='pt',
            truncation=True,
            padding=True,
            pad_to_multiple_of=32,
            max_length=model.config.max_position_embeddings)
        return result

    data = data.map(tokenization, batched = True)
    training_args = TRAININGCONFIG["fine-tuning"]["deberta-v3-large"]
    training_args.output_dir = str(BASE_PATH/"model")
    trainer = FineTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=data
    )
    trainer.train()
    trainer.save_model(BASE_PATH/"model"/target_class/f"deberta-v3-large-distill-al-{iteration+1}")

if __name__ == "__main__":
    main()