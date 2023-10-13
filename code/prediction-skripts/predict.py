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
    parser.add_argument("--task", type=str, default="zediac-large")
    args = parser.parse_args()

    target_class = args.target_class
    iteration = args.iteration

    model_size = args.task.split("-")[1]
    task = args.task.split("-")[0]

    if task == "zedi":
        model_path = BASE_PATH/"model"/target_class/f"deberta-v3-{model_size}-distill"
        out_path = BASE_PATH/"data"/"processed"/target_class/f"{args.dataset_name}-{task}-{model_size}"
        prediction_name = f"{target_class}_DI"
    elif task == "ft":
        model_path = BASE_PATH/"model"/target_class/f"deberta-v3-{model_size}-al"
        out_path = BASE_PATH/"data"/"processed"/target_class/f"{args.dataset_name}-{task}-{model_size}"
        prediction_name = f"{target_class}_FT"
    elif task == "zediac":
        model_path = BASE_PATH/"model"/target_class/f"deberta-v3-{model_size}-distill-al-{iteration}"
        out_path = BASE_PATH/"data"/"processed"/target_class/f"{args.dataset_name}-{task}-{model_size}-iteration-{iteration}"
        prediction_name = f"{target_class}_ZEDIAC_{iteration}"
        
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(f"microsoft/deberta-v3-{model_size}")
    
    training_args = TRAININGCONFIG["predict"]
    training_args.output_dir = "./model"
    trainer = Trainer(
        model = model,
        args = training_args
    )
    
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

    data = datasets.Dataset.load_from_disk(BASE_PATH/"data"/"preprocessed"/"dataset"/args.dataset_name)
    df = data.to_pandas()
    df = df.sort_values(by="n_words", ascending = False)
    data = datasets.Dataset.from_pandas(df, preserve_index=False)
    batch_size = 128
    data = data.map(tokenization, batched=True, batch_size=batch_size)
    data = data.map(predict_batch, batched=True, batch_size=batch_size)
    data.save_to_disk(out_path)
            
if __name__ == "__main__":
    main()
