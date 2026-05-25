#%%
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment
import shap
import pandas as pd
import numpy as np  
# %%
# Load prepred data
data = pd.read_csv("../data/processed/credit_risk_prepared.csv")
data.reset_index(inplace=True, drop=True)
data.head()
# %%
test_size, folds = .2, 10

model = setup(data = data, target = 'loan_status'
              , session_id=123
              ,train_size=(1-test_size)
              ,fold=folds
              ,) 

# %%
# Print the list of available models
best = compare_models(include = ['catboost','lda','lr','rf','et'])

# %%
# save pipeline
final_model = finalize_model(best)
save_model(best, '../models/automl_best_mdl')
# %%
