# Data Labeling

## Setup
 - navigate into this folder
 - run the following commands in the command line, make sure to have python >3.9 (you can replace ".lable-env" with whatever you like best)
    - `python -m venv .label-env`
    - `source .label-env/bin/activate`
    - `pip install --upgrade pip`
    - `pip install numpy pandas jupyterlab ipywidgets pigeonXT-jupyter python-dotenv`
    - `jupyter lab`

## Active Learning Labeling

 - I worked in batches/iterations of 500 lables and always first created a pickle file for labeling from the datasets in the data/processed folder (created either by distillation (iteration 0) or activel learning) using the `Initialize-new-Active-Learning-Annotation-Iteration.ipynb` notebook
 - After that I went to the `Active-Labeling.ipynb` notebook and adapted the iteration and labeled batches of 25 to 50 paragraphs with different strategies

## Evaluation Dataset

 - I first sampled the reports to be labeled for evaluation in the `Sampling-Eval-Dataset.ipynb` notebook
 - And then labelled as far as I got (i.e. 7 reports 5 of which 10Ks) in the `Labeling-Eval.ipynb` notebook

## Code Book

### Loss 
The statement describes or refers to a financial/business loss, or adverse business development experienced by the reporting entity.

### Unexpected
The statement describes of refers to an unexpected event or development experienced by or related to the reporting entity.

## Stats

The `Stats-for-Reporting.ipynb` notebook contains the code used to create overview tables in the final thesis