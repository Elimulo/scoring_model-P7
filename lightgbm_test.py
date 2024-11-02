# %% [code]
# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
import re
from contextlib import contextmanager
import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    end=time.time()
    print("{} - done in {:.0f}s".format(title, end - t0))



# LightGBM GBDT with KFold or Stratified KFold

from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, KFold
import mlflow
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import logging
logging.basicConfig(level=logging.INFO)
# logging.getLogger('mlflow').setLevel(logging.WARNING)

def custom_cost_metric(y_true, y_pred, fn_cost=10, fp_cost=1):
    """
    Custom metric that calculates the total cost based on false negatives (FN) and false positives (FP).
    
    Parameters:
    - y_true: Array of true binary labels
    - y_pred: Array of predicted probabilities
    - fn_cost: Cost associated with false negatives
    - fp_cost: Cost associated with false positives

    Returns:
    - total_cost: The computed cost
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost


def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # Define your parameters and log them with MLflow
    lgb_params = {
        'objective': 'binary',
        'nthread': 4,
        'n_estimators': 10000,
        'learning_rate': 0.02,
        'num_leaves': 25,
        'colsample_bytree': 0.9497036,
        'subsample': 0.8715623,
        'max_depth': 8,
        'reg_alpha': 0.041545473,
        'reg_lambda': 0.0735294,
        'min_split_gain': 0.0222415,
        'min_child_weight': 39.3259775,
        'metric': 'binary_logloss',
        'scale_pos_weight': 11.5 ,
         'verbose' : 1
        # 'is_unbalance' : True  ,
        # 'class_weight' : 'balanced',
   
    }

    # Log all parameters to MLflow
    for param, value in lgb_params.items():
        mlflow.log_param(param, value)

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print(f"Starting LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    del df
    gc.collect()

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        print(f"Fold {n_fold + 1}")

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf = LGBMClassifier(**lgb_params)
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=200)])

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        binary_preds = (oof_preds[valid_idx] > 0.5).astype(int)
        
        # Calculate and print metrics
        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        fold_f1 = f1_score(valid_y, binary_preds )
        fold_recall = recall_score(valid_y, binary_preds)
        fold_precision = precision_score(valid_y, binary_preds)
        
        # Calculate custom cost
        fold_custom_cost = custom_cost_metric(valid_y, oof_preds[valid_idx])
        
        print(f'Fold {n_fold + 1} AUC: {fold_auc:.6f}')
        print(f'Fold {n_fold + 1} F1 Score: {fold_f1:.6f}')
        print(f'Fold {n_fold + 1} Recall: {fold_recall:.6f}')
        print(f'Fold {n_fold + 1} Precision: {fold_precision:.6f}')
        print(f'Fold {n_fold + 1} Custom Cost: {fold_custom_cost:.6f}')
        
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(valid_y, (oof_preds[valid_idx] > 0.5).astype(int))
        print("Confusion Matrix:\n", cm)

        # Log metrics to MLflow
        mlflow.log_metric(f'fold_{n_fold + 1}_auc', fold_auc)
        mlflow.log_metric(f'fold_{n_fold + 1}_f1', fold_f1)
        mlflow.log_metric(f'fold_{n_fold + 1}_recall', fold_recall)
        mlflow.log_metric(f'fold_{n_fold + 1}_precision', fold_precision)
        mlflow.log_metric(f'fold_{n_fold + 1}_custom_cost', fold_custom_cost)

        # Log the model for the current fold
        mlflow.lightgbm.log_model(clf, f'model_fold_{n_fold + 1}')

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print(oof_preds.shape)
    print(binary_preds.shape)

    # Calculate and log overall metrics
    full_auc = roc_auc_score(train_df['TARGET'], oof_preds)
    # full_accuracy = accuracy_score(train_df['TARGET'], binary_preds)
    full_f1 = f1_score(train_df['TARGET'], binary_preds)
    full_recall = recall_score(train_df['TARGET'], binary_preds)
    full_precision = precision_score(train_df['TARGET'], binary_preds)
    full_custom_cost = custom_cost_metric(train_df['TARGET'], oof_preds)
    
    print(f'Full AUC score: {full_auc:.6f}')
    print(f'Full F1 Score: {full_f1:.6f}')
    print(f'Full Recall: {full_recall:.6f}')
    print(f'Full Precision: {full_precision:.6f}')
    print(f'Full Custom Cost: {full_custom_cost:.6f}')

    # Log the full metrics to MLflow
    mlflow.log_metric('full_auc', full_auc)
    mlflow.log_metric('full_accuracy', full_auc)
    mlflow.log_metric('full_f1', full_f1)
    mlflow.log_metric('full_recall', full_recall)
    mlflow.log_metric('full_precision', full_precision)
    mlflow.log_metric('full_custom_cost', full_custom_cost)

    # Log feature importance plot as an artifact
    display_importances(feature_importance_df)
    mlflow.log_artifact('lgbm_importances01.png')

    if not debug:
        test_df['TARGET'] = sub_preds
        submission_file_name = 'submission.csv'
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
        print(f'Submission file saved as {submission_file_name}')


def display_importances(feature_importance_df):
    """
    Function to display and save the feature importance plot.
    It assumes that feature importance data is passed as a DataFrame.
    """
    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:40].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(10, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug = True):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    num_rows = 10000 if debug else None
    mlflow.end_run()
    with mlflow.start_run(run_name="LightGBM with class weight balanced"):
        # Add experiment name and tags
        mlflow.set_tag("developer", "Elise")
        mlflow.set_tag("experiment", "LightGBM_kfold_3")
        
        df = pd.read_csv('data/final_df.csv')
        with timer("Run LightGBM with kfold"):
            # Démarrer une expérience MLflow
            feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= True, debug= debug)

if __name__ == "__main__":
    # submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
