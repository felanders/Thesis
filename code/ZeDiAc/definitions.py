from pathlib import Path

from transformers import TrainingArguments

MODELCONFIG = {
    "models": [
        "BART",
        "RoBERTa",
        "ALBERT",
        "DeBERTa",
        "LOSS",
        "UNEXPECTED"
    ],
    "BART": {
        "name": "facebook/bart-large-mnli",
        "abbrev": "B",
        "id2label": {
            "0": "contradiction",
            "1": "neutral",
            "2": "entailment"
        },
        "label2id": {
            "contradiction": 0,
            "entailment": 2,
            "neutral": 1
        }
    },
    "RoBERTa": {
        "name": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "abbrev": "R",
        "id2label": {
            "0": "entailment",
            "1": "neutral",
            "2": "contradiction"
        },
        "label2id": {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
    },
    "ALBERT": {
        "name": "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
        "abbrev": "A",
        "id2label": {
            "0": "entailment",
            "1": "neutral",
            "2": "contradiction"
        },
        "label2id": {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
    },
    "DeBERTa": {
        "name": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "abbrev": "D",
        "id2label": {
            "0": "entailment",
            "1": "neutral",
            "2": "contradiction"
        },
        "label2id": {
            "contradiction": 2,
            "entailment": 0,
            "neutral": 1
        }
    },
    "LOSS": {
        "name": "/cluster/home/afelderer/Thesis/models/deberta-v3-large-distill-loss",
        "model_architecture_name": "deberta-v3-large",
        "abbrev": "DL",
        "id2label": {
            "0": "loss",
            "1": "no loss"
        },
        "label2id": {
            "loss": 0,
            "no loss": 1
        }
    },
    "UNEXPECTED": {
        "name": "/cluster/home/afelderer/Thesis/models/deberta-v3-large-distill-unexpected",
        "model_architecture_name": "deberta-v3-large",
        "abbrev": "DU",
        "id2label": {
            "0": "unexpected",
            "1": "not unexpected"
        },
        "label2id": {
            "unexpected": 0,
            "not unexpected": 1
        }
    },
    "abbrev2model": {
        "A": "ALBERT",
        "R": "RoBERTa",
        "B": "BART",
        "D": "DeBERTa",
        "DL": "Distilled loss",
        "DU": "Distilled unexpected"
    },
}
LABELCONFIG = {
    "templates": {
        "This example is {}.": "T1",
        "The statement refers to {}.": "T2"
    },
    "labels": {
        "loss": {
            "a bussiness loss": "L1",
            "a loss": "L2",
            "an adverse development": "L3",
            "a business loss or adverse development": "L4"
        },
        "unexpected": {
            "an unexpected development": "U1",
            "a surprising development": "U2",
            "an unexpected or surprising development": "U3",
            "an unexpected event": "U4",
            "a surprising consequence": "U5"
        }
    }
}

BASE_PATH = Path("/cluster/scratch/afelderer/Thesis")

TRAININGCONFIG = {
    "predict": TrainingArguments(
        output_dir="",
        do_train=False,
        do_eval=False,
        do_predict=True,
        auto_find_batch_size=True,
    	disable_tqdm=True
    ),
    "distill": {
        "deberta-v3-large": TrainingArguments(
            output_dir="",
            do_train=True,
            do_eval=False,
            do_predict=False,
            gradient_accumulation_steps=4,
            learning_rate=8e-6,
            weight_decay=0.1,
            warmup_steps=500,
            save_total_limit=10,
            optim="adamw_torch",
            auto_find_batch_size=False,
            label_names=["true", "false"],
            report_to="tensorboard",
            save_strategy="epoch",
            logging_steps=500
        )
    },
    "fine-tuning": {
        "deberta-v3-large": TrainingArguments(
            output_dir="",
            do_train=True,
            do_eval=False,
            do_predict=False,
            gradient_accumulation_steps=2,
            learning_rate=8e-6,
            weight_decay=0.1,
            warmup_ratio=0, # this is for an alredy pretrained model
            save_total_limit=10,
            optim="adamw_torch",
            auto_find_batch_size=True,
            label_names=["true", "false"],
            #report_to="tensorboard",
            save_strategy="epoch",
            logging_steps=100,
            num_train_epochs=20
        )
    }
}
