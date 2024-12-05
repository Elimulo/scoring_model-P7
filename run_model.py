import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

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
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost

def train_model(train_df, name='LightGBM_Final'):
    
        
    feats=pd.read_csv('features.csv')
    feats=feats['feature'].tolist()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.end_run()
    with mlflow.start_run(run_name=name):


        train_x, train_y = train_df[feats], train_df['TARGET']

        X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


        params = {'class_weight':'balanced',
                'nthread': 4,
                'n_estimators': 10000,
                'learning_rate': 0.02,
                'num_leaves': 34,
                'colsample_bytree': 0.9497036,
                'subsample': 0.8715623,
                'max_depth': 8,
                'reg_alpha': 0.041545473,
                'reg_lambda': 0.0735294,
                'min_split_gain': 0.0222415,
                'min_child_weight': 39.3259775,
                'early_stopping_round' : 10,
            }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric='auc'
        )
        mlflow.lightgbm.log_model(model, name)
        # Evaluate on validation set
        y_proba=model.predict_proba(X_valid)
        y_pred = (y_proba[:, 0] <= 0.54).astype(int)

        accuracy = accuracy_score(y_valid, y_pred)
        roc_auc = roc_auc_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        custom_cost = custom_cost_metric(y_valid, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("custom_cost", custom_cost)

        print("Accuracy on validation set:", accuracy_score(y_valid, y_pred))
        print("Roc Auc score on validation set:", roc_auc_score(y_valid, y_pred))
        print("F1 score on validation set:", f1_score(y_valid, y_pred))
        print("Custom cost metric score on validation set:", custom_cost_metric(y_valid, y_pred))

        cm = confusion_matrix(y_valid, y_pred)
        # Create a heatmap for the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Predicted Class 0', 'Predicted Class 1'], 
                    yticklabels=['True Class 0', 'True Class 1'])

        # Titles and labels
        plt.title('Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.show()

        cm_image_path = "confusion_matrix.png"
        plt.savefig(cm_image_path)

        # Log confusion matrix image as artifact
        
        mlflow.set_tag("developer", "Elise")
        mlflow.set_tag("experiment", name)
    
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{name}"
        model_version = mlflow.register_model(model_uri, name)
        mlflow.log_artifact(cm_image_path)
        
        run_id=mlflow.active_run().info.run_id

        
    return run_id

# Main function execution
if __name__ == "__main__":
    train_df = pd.read_csv('data/train_df.csv')
    run_id = train_model(train_df)
    print(f"Run completed with run ID: {run_id}")