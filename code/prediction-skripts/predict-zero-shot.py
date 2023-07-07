import argparse
import logging
import torch
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import storage
from transformers import AutoModelForSequenceClassification, AutoTokenizer

storage_client = storage.Client()
bucket = storage_client.bucket("thesis-emerging-risk-data")

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["BART", "RoBERTa", "ALBERT", "DeBERTa"], type=str, default="BART")
    parser.add_argument("--xla", type=bool, default=False)
    parser.add_argument("--data_blob", type=str, default="paragraphs_clean_learn.csv")
    parser.add_argument("--config_blob", type=str, default="config.json")
    parser.add_argument("--save_every_multiplier", type=int, default=1)
    args = parser.parse_args()

    directory = f"""{args.model_name}_run_{datetime.now().strftime("%Y-%m-%d_%H-%M")}"""
    os.mkdir(directory)
    logging.basicConfig(
        filename=f"""{directory}/run.log""",
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    
    logger.info(f"Args: {args}")
    if args.xla:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device(devkind="TPU")
        logger.info("Using XLA device")
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {device} device")

    # Get Configs
    with bucket.blob(args.config_blob).open("r") as f:
        config = json.load(f)
    model_name = config["models"][args.model_name]["name"]
    batch_size = config["models"][args.model_name]["batch_size"]

    # Load Model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        logger.error("Exception occurred", exc_info=True)
        logger.info("Fast Tokenizer not found, using slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model.to(device)
    model.eval()

    logger.info("Model loaded")
    # Set up the rest
    ids = [id for id in model.config.id2label]
    ids.sort()
    hypotheses = {}
    hypotheses_col_names = []
    i = 1
    for t in config["templates"]:
        for l in config["labels"]:
            hypotheses[f"h_{i}"] = t.format(l)
            for id in ids:
                hypotheses_col_names.append(f"h_{i}_{model.config.id2label[id]}")
            i+=1
    n_hypotheses = len(hypotheses)

    logger.info(f"Created {n_hypotheses} hypotheses from templates and labels")
    # Read data
    with bucket.blob(args.data_blob).open("r") as f:
        df = pd.read_csv(f)
    paragraphs = df.text.to_list()
    del df

    # Create Batches
    input_batches = []
    label_batches = []
    i = 0
    for p in paragraphs:
        for h in hypotheses:
            if i % batch_size == 0:
                input_batches.append([p])
                label_batches.append([hypotheses[h]])
            else:
                input_batches[i//batch_size].append(p)
                label_batches[i//batch_size].append(hypotheses[h])
            i += 1

    # Predict
    logger.info("Starting predictions")
    torch.no_grad()
    logit_batches = []
    save_every = (n_hypotheses*batch_size)*args.save_every_multiplier
    n_paragraphs = int(save_every*batch_size / n_hypotheses)
    for i, batch in enumerate(zip(input_batches, label_batches)):
        logger.info(f"Predicting batch {i+1}/{len(input_batches)}")
        tokenized_batch = tokenizer(
            batch[0], batch[1],
            return_tensors='pt',
            truncation='only_first',
            padding=True,
            pad_to_multiple_of = 128) # For TPUs always use multiples of 128
        logits = model(**tokenized_batch.to(device))[0].to("cpu")
        logit_batches.append(logits)
        if len(logit_batches) % save_every == 0:
            n_th_save = int(i//save_every)
            logger.info(f"Saving {save_every} batches for the {n_th_save+1}'th time")
            with open(f"""{directory}/logits_{n_th_save*n_paragraphs}-{(n_th_save+1)*n_paragraphs}.npy""", "wb") as f:
                np.save(f, torch.cat(logit_batches).detach().numpy().reshape(-1, len(hypotheses_col_names)))
            with bucket.blob(f"""{directory}/logits_{n_th_save*n_paragraphs}-{(n_th_save+1)*n_paragraphs}.npy""").open("wb") as f:
                np.save(f, torch.cat(logit_batches).detach().numpy().reshape(-1, len(hypotheses_col_names)))
            logit_batches = []
    # Last save
    n_th_save = int(i//save_every)
    with open(f"""{directory}/logits_{n_th_save*n_paragraphs}-{(n_th_save+1)*n_paragraphs}.npy""", "wb") as f:
        np.save(f, torch.cat(logit_batches).detach().numpy().reshape(-1, len(hypotheses_col_names)))
    with bucket.blob(f"""{directory}/logits_{n_th_save*n_paragraphs}-{(n_th_save+1)*n_paragraphs}.npy""").open("wb") as f:
        np.save(f, torch.cat(logit_batches).detach().numpy().reshape(-1, len(hypotheses_col_names)))
    logger.info("Done")

if __name__ == "__main__":
    main()
