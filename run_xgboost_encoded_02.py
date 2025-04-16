from sklearn.model_selection import train_test_split 
import optuna
from sklearn.metrics import accuracy_score
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
import helpers
import time
import argparse


# === Functions ===

def abs_mean(x):
    return(abs(x).mean())


def parse_args():
    parser = argparse.ArgumentParser(description='Run XGBoost on encoded matrix')
    parser.add_argument('-m', '--matrix', type=str, help='Path to encoded matrix')
    parser.add_argument('-o', '--outcomes', type=str, help='Path to binary vector')
    parser.add_argument('-f', '--features', type=str, help='Path to features')
    parser.add_argument('-p', '--prefix', type=str, help='Prefix to save output files')
    return  parser.parse_args() 

# === End of fucntions ===

# Get command line arguments
args = parse_args()

# Initiate variables
path_data_matrix = args.matrix # path/to/encoded_matrix.txt
path_data_outcomes = args.outcomes # path/to/binary_vektor.txt
path_data_features = args.features # path/to/features.txt
prefix = args.prefix # path/to/prefix to save output files
random_state = 123
test_size = 0.2
n = 20 # The number of top features to plot
ts = time.time()

# Check input
helpers.check_input([path_data_matrix, path_data_outcomes, path_data_features])

# Load data
print("Loading data...")
matrix = np.loadtxt(path_data_matrix, dtype=np.bool_)
outcomes = np.loadtxt(path_data_outcomes, dtype=np.bool_)
features = np.loadtxt(path_data_features, dtype=np.str_)

# Show data sizes
print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")
print(f"Outcomes shape: {outcomes.shape[0]}")
print(f"Features shape: {features.shape[0]}")

# Split data into train, test and valid ones
X_temp, X_test, y_temp, y_test = train_test_split(matrix, outcomes, 
                                                  test_size=0.65, 
                                                  random_state=random_state, 
                                                  stratify=outcomes)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                                                  test_size=0.4286, 
                                                  random_state=random_state, 
                                                  stratify=y_temp)

# Initiate optuna function
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',  
        'seed': random_state,
        'early_stopping_rounds': 10,
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'eta': trial.suggest_float('eta', 0.01, 0.3),  
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    # Initiate the classifier
    clf_xgb = xgb.XGBClassifier(**param)

    # Train the model
    print("Training the model...")
    clf_xgb.fit(X_train,
                y_train,
                verbose=True,
                eval_set=[(X_test, y_test)])
    
    # Evaluate the model
    y_pred_proba = clf_xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc

# Hyperparameter selection
study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params

# Add other parameters
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['seed'] = random_state
best_params['early_stopping_rounds'] = 10

# Train the final model
clf_xgb = xgb.XGBClassifier(**best_params)
print("Training the final model...")
clf_xgb.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=True)

# Save hyperparameters
helpers.save_dict(study.best_params, prefix + '.best.params.txt')

# Overfitting check
y_train_pred_proba = clf_xgb.predict_proba(X_train)[:, 1]
y_val_pred_proba = clf_xgb.predict_proba(X_val)[:, 1]
y_test_pred_proba = clf_xgb.predict_proba(X_test)[:, 1]
train_aucpr = roc_auc_score(y_train, y_train_pred_proba)
val_aucpr = roc_auc_score(y_val, y_val_pred_proba)
test_aucpr = roc_auc_score(y_test, y_test_pred_proba)
print(f"Train AUC-PR: {train_aucpr:.4f}, Val AUC-PR: {val_aucpr:.4f}, Test AUC-PR: {test_aucpr:.4f}")

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
auc = roc_auc_score(y_test, y_pred_proba)
cr = classification_report(y_test, y_pred, zero_division=0, target_names=["0", "1"])

# Save quality metrix into file
with open(prefix + ".qc.txt", "w") as fh:
    print("AUC:", auc, file=fh)
    print(cr, file=fh)

# Get feature importances and save into file
importances = clf_xgb.feature_importances_
feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
feature_importances.to_csv(prefix + '.importance.csv', header=False)

# Plot Top feature importances
top_features = feature_importances.head(n)
plt.figure(figsize=(10, 6))
plt.subplots_adjust(bottom=0.3)
plt.bar(top_features.index, top_features.values)
plt.xticks(rotation=90)
plt.savefig(prefix + '.feature_importance.png')
plt.clf()

# Initiate SHAP values explaner
explainer = shap.TreeExplainer(clf_xgb)

# Estimate SHAP values
shap_values = explainer.shap_values(X_train)

# Make barplot for 20 TOP SHAP values
shap.summary_plot(shap_values=shap_values, feature_names=features, plot_type="bar")
plt.savefig(prefix + ".shap.png")

# Count mean absolute values of SHAP values
print("Counting mean abs SHAP values...")
mean_shap_values = np.apply_along_axis(abs_mean, axis=0, arr=shap_values)
df1 = pd.DataFrame({"features" : features,
                   "mean_shap_values" : mean_shap_values})

# Sort data frame by descending SHAP values
df1.sort_values("mean_shap_values", inplace=True, ascending=False)

# Save SHAP values into file
df1.to_csv(prefix + ".shap.txt", sep = " ", index=False, header=False)

# Save trained model
joblib.dump(clf_xgb, prefix+".model.pkl")

# Show time elepsed
helpers.show_time_elepsed(ts)
