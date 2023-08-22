import logging

import torch
import datasets
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer)
from ZeDiAc.definitions import BASE_PATH, MODELCONFIG, TRAININGCONFIG
from ZeDiAc.Trainers import DistillTrainer, FineTuningTrainer

datasets.disable_caching()

class BASETrainerWrapper:
    def __init__(self, model_name, dataset_name, target_class, TrainerClass=Trainer, training_args=TRAININGCONFIG["predict"], prediction_name="predict"):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.target_class = target_class
        self.data_path = BASE_PATH/"data"/"processed"/self.target_class/dataset_name
        self.data = self.load_data(dataset_name)  
        self.data = self.data.sort_values(by="n_words")
        self.data = datasets.Dataset.from_pandas(self.data, preserve_index=False)
        training_args.output_dir = str(BASE_PATH/"logs"/"train"/self.target_class)
        self.trainer = TrainerClass(
            model=self.model,
            args=training_args
        )
        self.prediction_name = prediction_name

    def tokenization(self, inputs):
        result = self.tokenizer(
            inputs["text"],
            return_tensors='pt',
            truncation=True,
            padding=True,
            pad_to_multiple_of=32,
            max_length=self.tokenizer.model_max_length)
        return result

    def tokenize(self):
        # Delete previous tokenizations
        self.data = self.data.to_pandas()
        for col_name in ['input_ids','token_type_ids','attention_mask']:
            if col_name in self.data.columns:
                self.data.drop(col_name, axis=1, inplace=True)
        self.data = datasets.Dataset.from_pandas(self.data, preserve_index=False)
        self.data = self.data.map(lambda x: self.tokenization(x), batched=True, batch_size=32)

    def predict_batch(self, batch):
        data_batch = datasets.Dataset.from_dict(batch)
        predictions = self.trainer.predict(test_dataset=data_batch)
        return {f"logits_{self.prediction_name}": predictions.predictions}

    def predict(self):
        self.tokenize()
        print(f"Starting prediction")
        self.data = self.data.map(self.predict_batch, batched=True)
        print(f"Saving predictions")
        self.save_data()

    def load_data(self, dataset_name):
        if self.data_path.exists() and self.data_path.is_dir() and (self.data_path/"dataset_info.json").exists():
            in_data_path = self.data_path
        else:
            in_data_path = BASE_PATH/"data"/"preprocessed"/"dataset"/dataset_name
        return datasets.Dataset.load_from_disk(in_data_path)
    
    def save_data(self):
        self.data.save_to_disk(self.data_path)

    def save_trainer(self, suffix=""):
        self.trainer.save_model(BASE_PATH/"model"/self.target_class/f"{self.model_name}-{suffix}")

    def to_prediction_mode(self, training_args=TRAININGCONFIG["predict"]):
        training_args.output_dir = str(BASE_PATH/"logs"/"predict"/self.target_class/self.data_path.name)
        self.trainer.args = training_args

class ZSPredictor(BASETrainerWrapper):
    def __init__(self, model_name, dataset_name, hypotheses, target_class, prediction_name="", ensemble_temperature=2):
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            target_class=target_class,
            prediction_name=prediction_name
        )
        self.hypotheses = hypotheses
        self.current_hypothesis = None
        self.ensemble_temperature = ensemble_temperature

    def tokenization(self, inputs):
        text_list = inputs["text"]
        hypothesis_list = [self.current_hypothesis]*len(text_list)
        result = self.tokenizer(
            text_list,
            hypothesis_list,
            return_tensors='pt',
            truncation='only_first',
            padding=True,
            pad_to_multiple_of=32,
            max_length=self.tokenizer.model_max_length)
        return result

    def predict(self):
        for hypothesis in self.hypotheses:
            self.current_hypothesis = hypothesis
            ## :TODO: add model name to prediction name
            self.prediction_name = f"ZS_{self.hypotheses[hypothesis]}"
            super().predict()
            self.get_p()
        self.data = self.data.map(self.compute_ensemble, batched=True)
        self.save_data()
        
    def get_p(self):
        entail_id = self.model.config.label2id["entailment"]
        logit_column = f"logits_{self.prediction_name}"
        p_column = f"p_{self.prediction_name}"
        self.data = self.data.map(lambda x: {p_column: softmax(x[logit_column], axis=1)[:,entail_id]}, batched=True)

    def compute_ensemble(self, batch):
        ensemble = []
        logit_columns = [column for column in self.data.column_names if column.startswith("logits")]
        for column in logit_columns:
            logits = np.array(batch[column])
            soft_target = softmax(logits/self.ensemble_temperature, axis=1)
            ensemble.append(soft_target)
        ensemble = np.array(ensemble).mean(axis=0)
        entail_id = self.model.config.label2id["entailment"]
        other_ids = [self.model.config.label2id[label] for label in ["neutral", "contradiction"]]
        return  {f"logits_ZS_ensemble_T_{self.ensemble_temperature}": np.column_stack([ensemble[:,entail_id], ensemble[:,other_ids].sum(axis=1)])}
    

class DistillationTrainer(BASETrainerWrapper):
    def __init__(self, model_name, dataset_name, target_class, downsampling_factor=3, prediction_name="distillation"):
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            target_class=target_class,
            TrainerClass=DistillTrainer,
            training_args=TRAININGCONFIG["distill"][model_name.split("/")[1]],
            prediction_name=prediction_name
        )
        self.downsampling_factor = downsampling_factor

    def downsample(self):
        d_t = self.data.filter(lambda ex: ex["labels"][0] >= 0.5)
        d_f = self.data.filter(lambda ex: ex["labels"][0] < 0.5)
        n_t = d_t.shape[0]
        n_f = d_f.shape[0]
        factor = n_f/n_t
        if factor > self.downsampling_factor:
            self.data = datasets.concatenate_datasets([d_t, d_f.train_test_split(test_size=d_t.shape[0]*self.downsampling_factor)["test"]], axis = 0)
            self.data = self.data.shuffle(seed=19950808)
        self.data = self.data.map(self.tokenization, batched=True, batch_size=128)

    def prepare_data(self):
        for column_name in self.data.column_names:
            if column_name.startswith("logits_ZS_ensemble_T_"):
                self.trainer.temperature = float(column_name.split("_")[-1])
                self.data = self.data.rename_column(column_name, "labels")
                self.downsample()
                return
        raise ValueError("No logits_ZS_ensemble_T found in data")
    
    def run(self):
        self.prepare_data()
        self.trainer.tokenizer = self.tokenizer
        self.trainer.train_dataset = self.data
        self.trainer.train()
        # :TODO: check name
        print("saving trainer to", self.prediction_name)
        self.save_trainer(self.prediction_name)
