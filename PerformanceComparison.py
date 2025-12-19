"""
Quick comparison script - see how your current approach performs
vs the improved approach
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge

print("="*70)
print("PERFORMANCE COMPARISON: YOUR CURRENT APPROACH VS IMPROVED")
print("="*70)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('train.csv')
print(f"Training samples: {len(train_df)}")

# ========================================
# TEST 1: YOUR CURRENT APPROACH
# ========================================
print("\n" + "="*70)
print("TEST 1: YOUR CURRENT PREPROCESSING APPROACH")
print("="*70)

try:
    from HousePricesRegressionHelpers import HousePricePreprocessor as CurrentPreprocessor
    from HousePricesRegressionHelpers import prepare_data as current_prepare_data
    
    # Prepare data with your current approach
    X_train_current, y_train_current = current_prepare_data('train.csv')
    
    # Check if target is log-transformed
    if y_train_current.max() > 1000:
        print("⚠ Warning: Target not log-transformed - transforming now...")
        y_train_current = np.log1p(y_train_current)
    
    # Initialize your current preprocessor
    current_preprocessor = CurrentPreprocessor()
    X_processed_current = current_preprocessor.fit_transform(X_train_current, y_train_current)
    
    print(f"Features after preprocessing: {X_processed_current.shape[1]}")
    
    # Test with multiple models
    models = {
        'Ridge': Ridge(alpha=10),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, 
            n_jobs=-1, verbose=-1
        )
    }
    
    current_scores = {}
    print("\nCurrent Approach CV Scores:")
    print("-" * 50)
    
    for name, model in models.items():
        scores = cross_val_score(
            model, X_processed_current, y_train_current,
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        rmse = np.sqrt(-scores.mean())
        std = np.sqrt(-scores).std()
        current_scores[name] = rmse
        print(f"{name:15s}: {rmse:.5f} (+/- {std:.5f})")
    
    current_avg = np.mean(list(current_scores.values()))
    print(f"\n{'Average':15s}: {current_avg:.5f}")
    
except Exception as e:
    print(f"Error with current approach: {e}")
    print("Skipping current approach test...")
    current_scores = None
    current_avg = None

# ========================================
# TEST 2: IMPROVED APPROACH
# ========================================
print("\n" + "="*70)
print("TEST 2: IMPROVED PREPROCESSING APPROACH")
print("="*70)

try:
    from HousePricesRegressionHelpers import HousePricePreprocessor as ImprovedPreprocessor
    from HousePricesRegressionHelpers import prepare_data as improved_prepare_data
    
    # Prepare data with improved approach
    X_train_improved, y_train_improved = improved_prepare_data('train.csv', log_transform_target=True)
    
    # Initialize improved preprocessor
    improved_preprocessor = ImprovedPreprocessor()
    X_processed_improved = improved_preprocessor.fit_transform(X_train_improved, y_train_improved)
    
    print(f"Features after preprocessing: {X_processed_improved.shape[1]}")
    
    improved_scores = {}
    print("\nImproved Approach CV Scores:")
    print("-" * 50)
    
    for name, model in models.items():
        scores = cross_val_score(
            model, X_processed_improved, y_train_improved,
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        rmse = np.sqrt(-scores.mean())
        std = np.sqrt(-scores).std()
        improved_scores[name] = rmse
        print(f"{name:15s}: {rmse:.5f} (+/- {std:.5f})")
    
    improved_avg = np.mean(list(improved_scores.values()))
    print(f"\n{'Average':15s}: {improved_avg:.5f}")
    
except Exception as e:
    print(f"Error with improved approach: {e}")
    print("Skipping improved approach test...")
    improved_scores = None
    improved_avg = None

# ========================================
# COMPARISON SUMMARY
# ========================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

if current_scores and improved_scores:
    print("\nModel-by-Model Comparison:")
    print("-" * 70)
    print(f"{'Model':<15} {'Current':<12} {'Improved':<12} {'Difference':<12} Status")
    print("-" * 70)
    
    for name in current_scores.keys():
        current = current_scores[name]
        improved = improved_scores[name]
        diff = current - improved
        pct = (diff / current) * 100
        
        if diff > 0:
            status = "✓ Better"
        elif diff < 0:
            status = "✗ Worse"
        else:
            status = "= Same"
        
        print(f"{name:<15} {current:.5f}    {improved:.5f}    {diff:+.5f}    {status}")
    
    print("-" * 70)
    avg_diff = current_avg - improved_avg
    avg_pct = (avg_diff / current_avg) * 100
    
    print(f"\n{'AVERAGE':<15} {current_avg:.5f}    {improved_avg:.5f}    {avg_diff:+.5f}")
    print(f"\nOverall Improvement: {avg_pct:+.2f}%")
    
    if avg_diff > 0:
        print("\n✓✓✓ IMPROVED APPROACH IS BETTER! ✓✓✓")
        print(f"The improved preprocessing reduces RMSE by {avg_diff:.5f}")
        print("This should translate to a better Kaggle leaderboard score!")
    elif avg_diff < -0.001:
        print("\n⚠ Current approach is performing better")
        print("You may want to keep your current approach or investigate why")
    else:
        print("\n≈ Performance is similar between approaches")

elif current_scores:
    print("\nYour current approach scores:")
    print(f"Average RMSE: {current_avg:.5f}")
    
elif improved_scores:
    print("\nImproved approach scores:")
    print(f"Average RMSE: {improved_avg:.5f}")

else:
    print("\nCould not run comparison - check that data files exist")

# ========================================
# RECOMMENDATIONS
# ========================================
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if current_scores and improved_scores:
    if avg_diff > 0.005:
        print("""
1. ✓ The improved approach shows significant improvement
2. Use the improved preprocessing for your final submission
3. Consider ensemble methods (averaging multiple models)
4. Run: python full_pipeline_with_submission.py
        """)
    elif avg_diff > 0:
        print("""
1. Improved approach shows slight improvement
2. Try both approaches and compare Kaggle leaderboard scores
3. Consider tuning hyperparameters further
4. Ensemble might give additional boost
        """)
    else:
        print("""
1. Your current approach is working well
2. Still worth trying improved features
3. Sometimes CV scores don't perfectly predict leaderboard
4. Consider testing both on Kaggle leaderboard
        """)
else:
    print("""
1. Make sure train.csv is in the current directory
2. Make sure both preprocessing files are available
3. Run this script again after fixing any import errors
    """)

print("\n" + "="*70)
print("Next steps:")
print("1. Review the scores above")
print("2. Run full_pipeline_with_submission.py to generate predictions")
print("3. Upload submission.csv to Kaggle")
print("4. Compare your leaderboard score with CV scores")
print("="*70)