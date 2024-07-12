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
import sys

def abs_mean(x):
    return(abs(x).mean())

path_data = sys.argv[1]
prefix = sys.argv[2]

df = pd.read_csv(path_data, sep=" ")
snp = [column for column in df if column != 'IID' and column != 'PHENOTYPE']

df_encoded = pd.get_dummies(df, columns=snp)
X = df_encoded.drop(['PHENOTYPE', 'IID'], axis=1).copy()
y = df_encoded['PHENOTYPE'].copy()
y = y.replace([1,2],[0,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.33)

clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42, eval_metric='aucpr', early_stopping_rounds=10)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            eval_set=[(X_test, y_test)])

svc_disp = RocCurveDisplay.from_estimator(clf_xgb, X_test, y_test)
plt.savefig(prefix + '.roc.png')

y_pred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=clf_xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,      
                              display_labels=['healhy','sick'])
disp.plot()
plt.savefig(prefix + '.confusion_matrix.png')

y_pred_proba = clf_xgb.predict_proba(X_test)[:, 1]

aucpr_score = roc_auc_score(y_test, y_pred_proba)

with open(prefix + ".qc.txt", "w") as fh:
    print("AUC-PR Score:", aucpr_score, file=fh)
    print(classification_report(y_test, y_pred), file=fh)

importances = clf_xgb.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
features.to_csv(prefix + '.importance.csv', header=False)

## n is amount of top snp on the chart
n = 20
groups = features.index[:n]
counts = features[:n]

plt.clf()
plt.figure(figsize=(10, 6))
plt.subplots_adjust(bottom=0.3)

plt.bar(groups, counts)
plt.xticks(rotation=90)
plt.savefig(prefix + '.feature_importance.png')

plt.clf()
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
