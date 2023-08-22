from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from pigeonXT import annotate
from scipy.special import softmax


class Annotator:
    """Annotator class for labeling data with pigeonXT"""
    def __init__(self, data, labels, out_file, text_column="text", batch_size=10):
        self.labels = labels
        if type(self.labels) == dict:
            self.short_labels = [self.labels[l] for l in self.labels]
        else:
            self.short_labels = self.labels
        self.out_path = Path(out_file)
        if self.out_path.exists():
            self.data = pd.read_pickle(self.out_path)
        else:
            self.data = data.to_pandas()
            for label in self.short_labels:
                if label not in self.data.columns:
                    self.data[label] = None
            if "labeled" not in self.data.columns:
                self.data["labeled"] = False
            if "strategy" not in self.data.columns:
                # Keep track how the sample was drawn
                self.data["strategy"] = None
            self.data.to_pickle(self.out_path)
        self.text_column = text_column
        self.batch_size = batch_size
        self.examples_in_file = len(self.data)
        self.current_df = self.data
    
    def get_unlabeled(self):
        return self.data[self.data.labeled == False]

    def sample(self, strategy="random"):
        """base sampling method randomly samples a batch of examples"""
        if strategy != "random":
            raise NotImplementedError
        unlabeled = self.get_unlabeled()
        self.current_indices = np.random.choice(
            unlabeled.index.tolist(), replace=False, size=self.batch_size)
        current_df = unlabeled.iloc[self.current_indices]
        current_df.loc[current_df.index,"strategy"] = strategy
        return current_df

    def display_fn(self, text):
        TEMPLATE = """<div style="background-color: rgb(244, 236, 216)"><div style="max-width:600px; word-wrap:break-word; font-size: 22px; line-height: 1.4; font-family: Palatino; padding: 50px">{text}</div></div>"""
        return display(HTML(TEMPLATE.format(**{"text": text.value})))

    def update_row(self, example, selected_labels):
        row_labels = np.zeros([len(self.labels)], dtype=np.uint8)
        row_indices = [self.short_labels.index(y) for y in selected_labels]
        row_labels[row_indices] = 1
        self.current_df.loc[self.current_df[self.text_column]
                            == example, self.short_labels] = row_labels
        self.current_df.loc[self.current_df[self.text_column]
                            == example, ["labeled"]] = True

    def final_processing(self, annotations):
        self.data.loc[self.current_df.index] = self.current_df
        self.data.to_pickle(self.out_path)

    def label_batch(self, strategy="random"):
        """starts the labeling process fro newly sampled batch"""
        self.current_df = self.sample(strategy=strategy)
        sentences = self.current_df[self.text_column].tolist()
        annotated = annotate(
            sentences,
            options=self.short_labels,
            task_type='multilabel-classification',
            buttons_in_a_row=3,
            reset_buttons_after_click=True,
            include_next=True,
            include_back=True,
            example_process_fn=self.update_row,
            final_process_fn=self.final_processing,
            display_fn=self.display_fn
        )

    def get_annotations_count_per_label(self):
        count_per_label = pd.DataFrame(
            columns=self.short_labels, index=['count'])
        for label in self.short_labels:
            count_per_label.loc['count', label] = len(
                self.data.loc[self.data[label] == 1.0])
        count_per_label["labeled_examples"] = self.data.labeled.sum()
        return count_per_label


class ActiveAnnotator(Annotator):
    def __init__(self, data, labels, out_file, target_label, target_true_id=0, text_column="text", batch_size=10):
        super().__init__(data=data, labels=labels, out_file=out_file,
                         text_column=text_column, batch_size=batch_size)
        self.target_label = target_label
        self.true_id = target_true_id
        for col_name in self.data.columns:
            if col_name.startswith("logits"):
                if f"""p_{"_".join(col_name.split("_")[1:])}""" not in self.data.columns:
                    self.data[f"""p_{"_".join(col_name.split("_")[1:])}"""] = self.data[col_name].apply(lambda x: softmax(x)[self.true_id])
        self.data.to_pickle(self.out_path)

    def sample(self, strategy="lc", batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if strategy == "random":
            return super().sample()
        else:
            unlabeled = self.get_unlabeled()
            p = np.array(unlabeled[f"p_{self.target_label}"].tolist())
            confidences = 0.5 - np.abs(p - 0.5)
            if strategy == "lc":
                # Least confident
                self.current_indices = np.argpartition(
                    confidences, -self.batch_size)[-self.batch_size:]
                current_df = unlabeled.iloc[self.current_indices]
            elif strategy == "hc_t":
                # Highest confidence for the true class
                self.current_indices = np.argpartition(
                    p, -self.batch_size)[-self.batch_size:]
                current_df = unlabeled.iloc[self.current_indices]
            elif strategy == "mix_lc_hc_t":
                # Mix of least confident and highest confidence for the true class
                if self.batch_size % 2 != 0:
                    raise ValueError("Batch size must be even")
                lc = np.argpartition(
                    confidences, -self.batch_size)[-self.batch_size:]
                hc = np.argpartition(p, -self.batch_size)[-self.batch_size:]
                self.current_indices = np.concatenate(
                    [lc[:self.batch_size//2], hc[:self.batch_size//2]])
                current_df = unlabeled.iloc[self.current_indices]
            current_df.loc[current_df.index,"strategy"] = f"{strategy}_{self.target_label}"
            return current_df

class ReportAnnotator(Annotator):
    def __init__(self, data, labels, out_file, text_column="text", batch_size=10):
        super().__init__(data=data, labels=labels, out_file=out_file,
                         text_column=text_column, batch_size=batch_size)
        self.data.to_pickle(self.out_path)

    def get_unlabeled(self):
        return self.data[self.data.labeled == -1]

    def sample(self, strategy="sequential"):
        if strategy == "random":
            return super().sample()
        current_df = self.get_unlabeled()
        self.current_indices = current_df.iloc[:self.batch_size].index
        current_df = current_df.loc[self.current_indices]
        current_df.loc[self.current_indices,"strategy"] = f"{strategy}"
        print("Companies: ", current_df.company.unique(), "Years: ", current_df.year.unique(), "Paragraphs:", current_df.paragraph_nr.unique()[0], "to", current_df.paragraph_nr.unique()[-1])
        return current_df
