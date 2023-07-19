import sys

sys.path.append("/cluster/home/afelderer/Thesis/code")

import argparse

import datasets
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer)
from ZeDiAc.definitions import BASE_PATH, TRAININGCONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", "-t", choices=["loss", "unexpected"], type=str, default="loss")
    parser.add_argument("--iteration", "-i", type=int, default=3)
    parser.add_argument("--dataset_name", "-d", type=str, default="evaluate")
    args = parser.parse_args()

    target_class = args.target_class
    iteration = args.iteration
    
    model = AutoModelForSequenceClassification.from_pretrained(BASE_PATH/"model"/target_class/f"deberta-v3-large-distill-al-{iteration}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH/"model"/target_class/"deberta-v3-large-distill")
    training_args = TRAININGCONFIG["predict"]
    training_args.output_dir = "./model"
    trainer = Trainer(
        model = model,
        args = training_args
    )
    prediction_name = f"{target_class}_AL_{iteration}"

    def tokenization(inputs):
        result = tokenizer(
            inputs["text"],
            return_tensors='pt',
            truncation=True,
            padding=True,
            pad_to_multiple_of=32,
            max_length=model.config.max_position_embeddings)
        return result
        
    def predict_batch(batch):
        data_batch = datasets.Dataset.from_dict(batch)
        predictions = trainer.predict(test_dataset=data_batch)
        return {f"logits_{prediction_name}": predictions.predictions}

    dataset_name = args.dataset_name
    data = datasets.Dataset.load_from_disk(BASE_PATH/"data"/"preprocessed"/"dataset"/dataset_name)
    df = data.to_pandas()
    df = df.sort_values(by="n_words", ascending = False)
    data = datasets.Dataset.from_pandas(df, preserve_index=False)
    data = data.map(tokenization, batched = True)
    data = data.map(predict_batch, batched=True)
    data.save_to_disk(BASE_PATH/"data"/"processed"/target_class/f"{dataset_name}-iteration-{iteration}")
            
if __name__ == "__main__":
    main()
