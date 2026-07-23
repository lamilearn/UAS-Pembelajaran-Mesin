import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_shipping.csv")
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Cost matrix definitions (USD)
COST_FN = 50.0  # Cost of missed late delivery (customer churn, penalty)
COST_FP = 10.0  # Cost of unnecessary express handling for on-time delivery

def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    print("="*60)
    print("         MODEL TRAINING & CROSS-VALIDATION (src/train.py)")
    print("="*60)
    
    # 1. Drop identifier column (ID) to prevent leakage
    X = df.drop(columns=['ID', 'Reached.on.Time_Y.N'])
    y = df['Reached.on.Time_Y.N']
    
    # 2. Strict Train-Test Split BEFORE any preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"\n[DATA SPLIT]")
    print(f"   - Training Set : {X_train.shape[0]} samples")
    print(f"   - Test Set     : {X_test.shape[0]} samples (Saved to data/test_data.csv)")
    
    # Save test data for single-pass evaluation in src/evaluate.py
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    # 3. Identify feature types
    numeric_features = [
        'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
        'Prior_purchases', 'Discount_offered', 'Weight_in_gms'
    ]
    categorical_features = [
        'Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'
    ]
    
    # 4. Define Leak-Free Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 5. Model Candidates
    candidate_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['f1', 'roc_auc', 'precision', 'recall', 'accuracy']
    
    results = {}
    best_model_name = None
    best_cv_f1 = -1.0
    
    print("\n--- 5-FOLD CROSS-VALIDATION RESULTS ---")
    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        
        f1_mean, f1_std = np.mean(scores['test_f1']), np.std(scores['test_f1'])
        roc_mean, roc_std = np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc'])
        prec_mean = np.mean(scores['test_precision'])
        rec_mean = np.mean(scores['test_recall'])
        
        results[name] = {
            'f1_mean': float(f1_mean),
            'f1_std': float(f1_std),
            'roc_auc_mean': float(roc_mean),
            'roc_auc_std': float(roc_std),
            'precision_mean': float(prec_mean),
            'recall_mean': float(rec_mean)
        }
        
        print(f"\n> Algorithm: {name}")
        print(f"   - F1-Score : {f1_mean:.4f} (+/- {f1_std:.4f})")
        print(f"   - ROC-AUC  : {roc_mean:.4f} (+/- {roc_std:.4f})")
        print(f"   - Precision: {prec_mean:.4f} | Recall: {rec_mean:.4f}")
        
        if f1_mean > best_cv_f1:
            best_cv_f1 = f1_mean
            best_model_name = name
            
    print(f"\n[BEST MODEL SELECTED]: {best_model_name} (F1 = {best_cv_f1:.4f})")
    
    # 6. Fit final best pipeline on full training data
    best_model = candidate_models[best_model_name]
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    best_pipeline.fit(X_train, y_train)
    
    # 7. Threshold Tuning for Business Cost Minimization
    train_probs = best_pipeline.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = []
    
    for t in thresholds:
        preds = (train_probs >= t).astype(int)
        fn = np.sum((y_train == 1) & (preds == 0))
        fp = np.sum((y_train == 0) & (preds == 1))
        total_cost = (fn * COST_FN) + (fp * COST_FP)
        costs.append(total_cost)
        
    best_idx = np.argmin(costs)
    optimal_threshold = float(thresholds[best_idx])
    min_train_cost = float(costs[best_idx])
    
    print(f"\n[THRESHOLD OPTIMIZATION]")
    print(f"   - Default Threshold (0.50) Cost: ${costs[40]:,.2f}")
    print(f"   - Optimal Threshold ({optimal_threshold:.2f}) Cost: ${min_train_cost:,.2f}")
    print(f"   - Cost Savings: ${costs[40] - min_train_cost:,.2f}")
    
    # 8. Save Full Pipeline to models/model.joblib
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\n[SAVED ARTIFACT] Full sklearn Pipeline -> {model_path}")
    
    # 9. Save Metadata to models/metadata.json
    metadata = {
        "best_model": best_model_name,
        "cv_results": results,
        "optimal_threshold": optimal_threshold,
        "cost_matrix": {"COST_FN": COST_FN, "COST_FP": COST_FP},
        "feature_names": {
            "numeric": numeric_features,
            "categorical": categorical_features
        },
        "target_classes": {
            "0": "tepat_waktu",
            "1": "terlambat"
        }
    }
    
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[SAVED ARTIFACT] Metadata -> {metadata_path}")
    print("\n" + "="*60)
    print("Training pipeline finished successfully!")
    print("="*60)

if __name__ == "__main__":
    run_training()
