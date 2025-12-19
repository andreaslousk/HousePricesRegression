"""
Complete pipeline for Kaggle House Prices Competition
Trains model, makes predictions, and generates submission file
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

# Import your preprocessor
from HousePricesRegressionHelpers import HousePricePreprocessor, prepare_data


def train_and_predict_single_model(X_train, y_train, X_test, model_name='xgboost'):
    """
    Train a single model and make predictions.
    
    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_test: Test features
        model_name: 'xgboost', 'lightgbm', 'ridge', etc.
    
    Returns:
        predictions: Test predictions (in original scale)
        cv_score: Cross-validation RMSE
    """
    
    # Initialize preprocessor
    preprocessor = HousePricePreprocessor()
    
    # Preprocess training data
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Select model
    if model_name == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,  # Slower learning
            max_depth=3,         # Shallower trees
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.05,      # L1 regularization
            reg_lambda=0.05,     # L2 regularization
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'lightgbm':
        model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_name == 'ridge':
        model = Ridge(alpha=15)  # More regularization
    elif model_name == 'lasso':
        model = Lasso(alpha=0.001, max_iter=10000)
    elif model_name == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=2000,
            learning_rate=0.01,  # Slower learning
            max_depth=3,         # Shallower trees
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.7,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Cross-validation score
    print(f"\nEvaluating {model_name} with 10-fold CV...")
    cv_scores = cross_val_score(
        model, X_train_processed, y_train,
        cv=10, scoring='neg_mean_squared_error', n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_std = np.sqrt(-cv_scores).std()
    
    print(f"{model_name} CV RMSE: {cv_rmse:.5f} (+/- {cv_std:.5f})")
    
    # Train on full training data
    print(f"Training {model_name} on full training set...")
    model.fit(X_train_processed, y_train)
    
    # Make predictions
    print("Making predictions on test set...")
    y_test_pred_log = model.predict(X_test_processed)
    
    # Transform back to original scale
    y_test_pred = np.expm1(y_test_pred_log)
    
    return y_test_pred, cv_rmse, model, preprocessor


def train_and_predict_ensemble(X_train, y_train, X_test, models_to_use=None):
    """
    Train multiple models and average their predictions.
    This often gives the best results!
    
    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        X_test: Test features
        models_to_use: List of model names (default: uses best 4)
    
    Returns:
        predictions: Ensemble test predictions
        cv_scores: Dictionary of CV scores for each model
    """
    
    if models_to_use is None:
        models_to_use = ['xgboost', 'lightgbm', 'ridge', 'gbm']
    
    print("="*60)
    print("TRAINING ENSEMBLE OF MODELS")
    print("="*60)
    
    all_predictions = []
    cv_scores = {}
    
    for model_name in models_to_use:
        pred, cv_score, model, preprocessor = train_and_predict_single_model(
            X_train, y_train, X_test, model_name
        )
        all_predictions.append(pred)
        cv_scores[model_name] = cv_score
    
    # Average predictions
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    for name, score in cv_scores.items():
        print(f"{name}: {score:.5f}")
    print(f"\nAverage CV RMSE: {np.mean(list(cv_scores.values())):.5f}")
    
    return ensemble_predictions, cv_scores


def create_submission_file(predictions, test_ids, filename='submission.csv'):
    """
    Create a submission file in Kaggle format.
    
    Args:
        predictions: Array of predictions
        test_ids: Array of test IDs
        filename: Output filename
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"\nSubmission file saved as: {filename}")
    print(f"Shape: {submission.shape}")
    print(f"\nFirst few predictions:")
    print(submission.head(10))
    print(f"\nPrediction statistics:")
    print(submission['SalePrice'].describe())
    
    return submission


def compare_with_old_approach(X_train, y_train, X_test):
    """
    Compare your old preprocessing approach vs the improved one.
    This helps you see if the changes actually improve performance!
    """
    print("\n" + "="*60)
    print("COMPARING OLD VS NEW PREPROCESSING")
    print("="*60)
    
    # Try to import your old preprocessor
    try:
        from HousePricesRegressionHelpers import HousePricePreprocessor as OldPreprocessor
        
        print("\nTesting OLD preprocessor...")
        old_preprocessor = OldPreprocessor()
        X_train_old = old_preprocessor.fit_transform(X_train, y_train)
        
        model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        cv_scores_old = cross_val_score(
            model, X_train_old, y_train,
            cv=5, scoring='neg_mean_squared_error'
        )
        old_rmse = np.sqrt(-cv_scores_old.mean())
        print(f"OLD approach CV RMSE: {old_rmse:.5f}")
        
    except ImportError:
        print("Could not import old preprocessor - skipping comparison")
        old_rmse = None
    
    # Test NEW preprocessor
    print("\nTesting NEW preprocessor...")
    new_preprocessor = HousePricePreprocessor()
    X_train_new = new_preprocessor.fit_transform(X_train, y_train)
    
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    cv_scores_new = cross_val_score(
        model, X_train_new, y_train,
        cv=5, scoring='neg_mean_squared_error'
    )
    new_rmse = np.sqrt(-cv_scores_new.mean())
    print(f"NEW approach CV RMSE: {new_rmse:.5f}")
    
    if old_rmse is not None:
        improvement = old_rmse - new_rmse
        pct_improvement = (improvement / old_rmse) * 100
        print(f"\nImprovement: {improvement:.5f} ({pct_improvement:.2f}%)")
        
        if improvement > 0:
            print("✓ New approach is BETTER!")
        else:
            print("✗ Old approach was better - may need to adjust")


# ========================================
# MAIN PIPELINE
# ========================================
def main(train_path='train.csv', test_path='test.csv', approach='ensemble'):
    """
    Main pipeline to train and predict.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        approach: 'single' or 'ensemble'
    """
    
    print("="*60)
    print("KAGGLE HOUSE PRICES - FULL PIPELINE")
    print("="*60)
    
    # Load training data
    print("\nLoading training data...")
    X_train, y_train = prepare_data(train_path, log_transform_target=True)
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Remove additional outliers
    print("\nRemoving outliers...")
    original_size = len(X_train)
    mask = ~((X_train['GrLivArea'] > 4000) & (y_train < np.log1p(300000)))
    X_train = X_train[mask]
    y_train = y_train[mask]
    print(f"Removed {original_size - len(X_train)} outliers")
    print(f"Training samples after outlier removal: {len(X_train)}")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(test_path)
    test_ids = test_df['Id'].values
    
    X_test, _ = prepare_data(test_path, log_transform_target=False)
    print(f"Test samples: {len(X_test)}")
    
    # Compare approaches (if old file exists)
    # compare_with_old_approach(X_train, y_train, X_test)
    
    # Make predictions
    if approach == 'single':
        print("\n" + "="*60)
        print("TRAINING SINGLE MODEL (XGBoost)")
        print("="*60)
        predictions, cv_score, model, preprocessor = train_and_predict_single_model(
            X_train, y_train, X_test, model_name='xgboost'
        )
        filename = 'submission_xgboost.csv'
        
    elif approach == 'ensemble':
        predictions, cv_scores = train_and_predict_ensemble(
            X_train, y_train, X_test
        )
        filename = 'submission_ensemble.csv'
    
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    # Create submission file
    print("\n" + "="*60)
    print("CREATING SUBMISSION FILE")
    print("="*60)
    submission = create_submission_file(predictions, test_ids, filename)
    
    print("\n" + "="*60)
    print("DONE! Upload your submission to Kaggle.")
    print("="*60)
    
    return submission, predictions


# ========================================
# QUICK TEST FUNCTIONS
# ========================================
def quick_cv_test(train_path='train.csv'):
    """
    Quick function to just see CV scores without making predictions.
    Useful for rapid experimentation.
    """
    print("Loading data...")
    X_train, y_train = prepare_data(train_path, log_transform_target=True)
    
    models_to_test = {
        'Ridge': Ridge(alpha=10),
        'XGBoost': xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, 
                                     max_depth=4, subsample=0.8, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42,
                                       verbose=-1)
    }
    
    preprocessor = HousePricePreprocessor()
    X_processed = preprocessor.fit_transform(X_train, y_train)
    
    print("\nQuick CV Test Results:")
    print("-" * 40)
    
    for name, model in models_to_test.items():
        scores = cross_val_score(model, X_processed, y_train, cv=5,
                                scoring='neg_mean_squared_error', n_jobs=-1)
        rmse = np.sqrt(-scores.mean())
        print(f"{name:15s}: {rmse:.5f}")


if __name__ == "__main__":
    # You can run different approaches:
    
    # Option 1: Quick CV test (no predictions, just see scores)
    # quick_cv_test('train.csv')
    
    # Option 2: Single model (XGBoost)
    # submission, predictions = main('train.csv', 'test.csv', approach='single')
    
    # Option 3: Ensemble (RECOMMENDED - usually best results)
    submission, predictions = main('train.csv', 'test.csv', approach='ensemble')
    
    # The submission file is now ready to upload to Kaggle!c