""" 
The script trains XGBoost model over genotypes of diploid organism to predict the binary outcome. 
It also ranks the single nucleotide polymorphisms (SNPs) by feature importance and SHAP values. 
The input is a tabular data with SNPs in columns and individuals in rows. 
There are also columns IID with sample IDs and PHENOTYPE with {1, 2} for control and case samples, respectively.  
They are transformed into {0, 1} for control and case samples.
The genotypes are assumed to be categorical. The SNPs are firstly encoded with one-hot algorithm.  
Author: Mikhail Belousov
"""

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import RocCurveDisplay 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import xgboost as xgb 
import shap 
import os, sys
import helpers
import time

# === Functions ===

def abs_mean(x):
    return(abs(x).mean())

# === End of fucntions ===

# Get command line arguments
path_data = sys.argv[1] # path/to/genotypes.txt
prefix = sys.argv[2] # path/to/prefix to save output files

# Check input
helpers.check_input([path_data])

# Initiate variables
ts = time.time()

# Load input dataset
df = pd.read_csv(path_data, sep=" ")
print("Shape of input data:", df.shape)

# Get IDs of features
snp = [column for column in df if column != 'IID' and column != 'PHENOTYPE']

# One-hot encode features
df_encoded = pd.get_dummies(df, columns=snp)

# Subset matrix with features
X = df_encoded.drop(['PHENOTYPE', 'IID'], axis=1).copy()

# Define outcome variable
y = df_encoded['PHENOTYPE'].copy()
y = y.replace([1,2],[0,1])

# Split data into train and test ones
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.33)

# Initiate the classifier
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42, eval_metric='aucpr', early_stopping_rounds=10)

# Train the model
print("Training the model")
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            eval_set=[(X_test, y_test)])

# Plot ROC curve
svc_disp = RocCurveDisplay.from_estimator(clf_xgb, X_test, y_test)
plt.savefig(prefix + '.roc.png')

# Get and plot confusion matrix
y_pred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=clf_xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,      
                              display_labels=['healhy','sick'])
disp.plot()
plt.savefig(prefix + '.confusion_matrix.png')
plt.clf()

# Estimate the quality of model
y_pred_proba = clf_xgb.predict_proba(X_test)[:, 1]
aucpr_score = roc_auc_score(y_test, y_pred_proba)

# Save quality metrix into file
with open(prefix + ".qc.txt", "w") as fh:
    print("AUC-PR Score:", aucpr_score, file=fh)
    print(classification_report(y_test, y_pred), file=fh)

# Get feature importances and save into file
importances = clf_xgb.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
features.to_csv(prefix + '.importance.csv', header=False)

# Plot Top 20 feature importances
n = 20
groups = features.index[:n]
counts = features[:n]

plt.figure(figsize=(10, 6))
plt.subplots_adjust(bottom=0.3)
plt.bar(groups, counts)
plt.xticks(rotation=90)
plt.savefig(prefix + '.feature_importance.png')

plt.clf()

# Initiate SHAP values explaner
explainer = shap.TreeExplainer(clf_xgb)

# Estimate SHAP values
shap_values = explainer.shap_values(X_train)

# Make barplot for 20 TOP SHAP values
shap.summary_plot(shap_values=shap_values, feature_names=X.columns, plot_type="bar")
plt.savefig(prefix + ".shap.png")

# Count mean absolute values of SHAP values
print("Counting mean abs SHAP values...")
mean_shap_values = np.apply_along_axis(abs_mean, axis=0, arr=shap_values)
df1 = pd.DataFrame({"features" : X.columns,
                   "mean_shap_values" : mean_shap_values})

# Sort data frame by descending SHAP values
df1.sort_values("mean_shap_values", inplace=True, ascending=False)

# Save SHAP values into file
df1.to_csv(prefix + ".shap.txt", sep = " ", index=False, header=False)

# Show time elepsed
helpers.show_time_elepsed(ts)

