from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import RocCurveDisplay 
import joblib
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
path_data_matrix = sys.argv[1] # path/to/encoded_matrix.txt
path_data_outcomes = sys.argv[2] # path/to/binary_vektor.txt
path_data_features = sys.argv[3] # path/to/features.txt
prefix = sys.argv[4] # path/to/prefix to save output files


# Check input
helpers.check_input([path_data_matrix])
helpers.check_input([path_data_outcomes])
helpers.check_input([path_data_features])


# Initiate variables
ts = time.time()

# Loading data
matrix = pd.read_csv(path_data_matrix, sep=" ")
with open(path_data_outcomes, 'r') as file:
    outcomes = list(map(int, file.read().split()))
with open(path_data_features, 'r') as file:
    features = file.read().split()


outcomes = outcomes[0:-1]
matrix.columns =features


# Split data into train and test ones
X_train, X_test, y_train, y_test = train_test_split(matrix, outcomes, random_state=42, stratify=outcomes, test_size=0.33)

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
features = pd.Series(importances, index=matrix.columns).sort_values(ascending=False)
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
shap.summary_plot(shap_values=shap_values, feature_names=matrix.columns, plot_type="bar")
plt.savefig(prefix + ".shap.png")

# Count mean absolute values of SHAP values
print("Counting mean abs SHAP values...")
mean_shap_values = np.apply_along_axis(abs_mean, axis=0, arr=shap_values)
df1 = pd.DataFrame({"features" : matrix.columns,
                   "mean_shap_values" : mean_shap_values})

# Sort data frame by descending SHAP values
df1.sort_values("mean_shap_values", inplace=True, ascending=False)

# Save SHAP values into file
df1.to_csv(prefix + ".shap.txt", sep = " ", index=False, header=False)

# Save trained model
joblib.dump(clf_xgb, prefix+".model.pkl")

# Show time elepsed
helpers.show_time_elepsed(ts)
