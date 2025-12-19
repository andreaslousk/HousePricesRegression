import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures

class HousePricePreprocessor(BaseEstimator, TransformerMixin):
    """
    Improved preprocessor for house price prediction.
    """
    
    def __init__(self, use_polynomial=False):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.categorical_features = None
        self.numerical_features = None
        self.max_garage_age = None
        self.use_polynomial = use_polynomial
        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True) if use_polynomial else None
        
    def fit(self, X, y=None):
        """Fit the preprocessor on training data."""
        X = X.copy()
        
        # Identify feature types
        self.categorical_features = X.select_dtypes(["object"]).columns.tolist()
        
        # Engineer features first
        X = self._engineer_features(X, fit=True)
        
        # Now identify numerical features (after engineering)
        self.numerical_features = X.select_dtypes([np.number]).columns.tolist()
        
        # Fit scaler on all features
        self.scaler.fit(X[self.numerical_features])
        
        # Fit polynomial features if used
        if self.use_polynomial:
            X_scaled = self.scaler.transform(X[self.numerical_features])
            self.poly.fit(X_scaled)
        
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessor."""
        X = X.copy()
        
        # Engineer features
        X = self._engineer_features(X, fit=False)
        
        # Scale numerical features
        X_scaled = self.scaler.transform(X[self.numerical_features])
        
        # Add polynomial features if enabled
        if self.use_polynomial:
            X_poly = self.poly.transform(X_scaled)
            feature_names = self.poly.get_feature_names_out(self.numerical_features)
            return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        return pd.DataFrame(X_scaled, columns=self.numerical_features, index=X.index)
    
    def _engineer_features(self, X, fit=False):
        """Apply feature engineering with more meaningful interactions."""
        
        # Calculate all new features in a dictionary first
        features = {}
        
        # === BATHROOM FEATURES ===
        total_bath = (X["BsmtFullBath"] + X["FullBath"] + 
                     0.5*X["BsmtHalfBath"] + 0.5*X["HalfBath"])
        features["TotalBath"] = total_bath
        features["BathPerBedroom"] = total_bath / (X["BedroomAbvGr"] + 1)
        
        # === AGE FEATURES ===
        features["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        features["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        garage_age = X["YrSold"] - X["GarageYrBlt"]
        
        # Fill GarageAge NaN
        if fit:
            self.max_garage_age = garage_age.max()
        features["GarageAge"] = garage_age.fillna(self.max_garage_age)
        features["HasGarage"] = (garage_age < self.max_garage_age).astype(int)
        features["IsRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)
        features["YearsSinceRemod"] = X["YrSold"] - X["YearRemodAdd"]
        
        # === AREA FEATURES ===
        total_sf = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        total_porch = (X["OpenPorchSF"] + X["EnclosedPorch"] + 
                      X["3SsnPorch"] + X["ScreenPorch"] + X["WoodDeckSF"])
        
        features["TotalSF"] = total_sf
        features["TotalPorchSF"] = total_porch
        features["Has2ndFloor"] = (X["2ndFlrSF"] > 0).astype(int)
        features["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)
        features["HasPool"] = (X["PoolArea"] > 0).astype(int)
        features["HasPorch"] = (total_porch > 0).astype(int)
        features["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        features["LivingAreaQuality"] = X["GrLivArea"] * X["OverallQual"]
        features["SFPerRoom"] = X["GrLivArea"] / (X["TotRmsAbvGrd"] + 1)
        
        # === LOT FEATURES ===
        features["LotArea_x_Frontage"] = X["LotArea"] * X["LotFrontage"]
        features["HasLotFrontage"] = (X["LotFrontage"] > 0).astype(int)
        
        # === GARAGE FEATURES ===
        features["GarageScore"] = X["GarageCars"] * X["GarageArea"]
        features["GarageCarPerArea"] = X["GarageCars"] / (X["GarageArea"] + 1)
        
        # === KITCHEN & ROOM FEATURES ===
        features["RoomPerArea"] = X["TotRmsAbvGrd"] / X["GrLivArea"]
        
        # === BASEMENT FEATURES ===
        features["BsmtFinishedRatio"] = X["BsmtFinSF1"] / (X["TotalBsmtSF"] + 1)
        features["BsmtScore"] = X["BsmtFinSF1"] + X["BsmtFinSF2"]
        
        # === QUALITY INTERACTIONS ===
        features["OverallScore"] = X["OverallQual"] * X["OverallCond"]
        features["QualityTimesArea"] = X["OverallQual"] * X["GrLivArea"]
        features["GarageScore2"] = X["GarageArea"] * X["OverallQual"]
        
        # === SEASON/TIME FEATURES ===
        features["SeasonSold"] = X["MoSold"].map({12:0, 1:0, 2:0,  # Winter
                                                   3:1, 4:1, 5:1,   # Spring
                                                   6:2, 7:2, 8:2,   # Summer
                                                   9:3, 10:3, 11:3}) # Fall
        
        # Columns to keep from original (not used in feature engineering)
        keep_cols = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond",
                     "MasVnrArea", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                     "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "TotalBsmtSF",
                     "HeatingQC", "CentralAir", "GrLivArea", "BedroomAbvGr", "KitchenAbvGr",
                     "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "FireplaceQu",
                     "GarageCars", "GarageArea", "GarageQual", "GarageCond", "PavedDrive",
                     "WoodDeckSF", "PoolArea", "PoolQC", "Fence", "MiscFeature", "MiscVal",
                     "MoSold", "YrSold", "SaleType", "SaleCondition"]
        
        # Filter to only columns that exist in X
        keep_cols = [col for col in keep_cols if col in X.columns]
        
        # Build new dataframe once
        X_new = pd.DataFrame(features, index=X.index)
        X_final = pd.concat([X[keep_cols], X_new], axis=1)
        
        return X_final


def prepare_data(filepath, log_transform_target=True):
    """Load and prepare data for modeling."""
    df = pd.read_csv(filepath).drop(columns=["Id"])
    
    # Get feature types
    categorical_features = df.select_dtypes(["object"]).columns
    
    # === HANDLE MISSING VALUES ===
    # Categorical - None means "doesn't have this feature"
    df[categorical_features] = df[categorical_features].fillna("None")
    
    # Numerical - strategic fills
    df["LotFrontage"] = df["LotFrontage"].fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["GarageCars"] = df["GarageCars"].fillna(0)
    df["GarageArea"] = df["GarageArea"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])  # Use house year if no garage
    
    # Basement features
    basement_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", 
                     "BsmtFullBath", "BsmtHalfBath"]
    df[basement_cols] = df[basement_cols].fillna(0)
    
    # === HANDLE OUTLIERS (based on common knowledge about this dataset) ===
    # Remove extreme outliers in GrLivArea
    if 'SalePrice' in df.columns:
        df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
    
    # === ENCODE CATEGORICAL FEATURES ===
    
    # Ordinal encoding for quality features
    quality_mapping = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'None': 0}
    quality_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
    ]
    
    # Other ordinal features with domain knowledge
    bsmt_exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
    bsmt_fin_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
    fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}
    garage_finish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
    paved_drive_map = {'Y': 2, 'P': 1, 'N': 0}
    utilities_map = {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}
    lot_shape_map = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
    land_slope_map = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
    functional_map = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
    central_air_map = {'Y': 1, 'N': 0}
    
    # Apply ordinal mappings
    ordinal_mappings = {
        'BsmtExposure': bsmt_exposure_map,
        'BsmtFinType1': bsmt_fin_map,
        'BsmtFinType2': bsmt_fin_map,
        'Fence': fence_map,
        'GarageFinish': garage_finish_map,
        'PavedDrive': paved_drive_map,
        'Utilities': utilities_map,
        'LotShape': lot_shape_map,
        'LandSlope': land_slope_map,
        'Functional': functional_map,
        'CentralAir': central_air_map
    }
    
    # Apply quality mappings
    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping)
    
    # Apply other ordinal mappings
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Frequency encoding for remaining categoricals
    remaining_categoricals = df.select_dtypes(["object"]).columns
    freq_features = {}
    for col in remaining_categoricals:
        freq_encoding = df[col].value_counts(normalize=True).to_dict()
        freq_features[f"{col}_freq"] = df[col].map(freq_encoding)
    
    # Add all frequency features at once
    if freq_features:
        df = pd.concat([df, pd.DataFrame(freq_features, index=df.index)], axis=1)
        # Drop original categorical columns
        df = df.drop(columns=remaining_categoricals)
    
    # Split features and target
    if 'SalePrice' in df.columns:
        X = df.drop("SalePrice", axis=1)
        y = df["SalePrice"]
        
        # Log transform target
        if log_transform_target:
            y = np.log1p(y)
        
        return X, y
    else:
        return df, None


def train_with_cv_target_encoding(X_train, y_train, model, n_splits=5):
    """
    Train with proper cross-validation to avoid target encoding leakage.
    Returns trained models and OOF predictions.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import TargetEncoder
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X_train))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Fit preprocessor on training fold only
        preprocessor = HousePricePreprocessor()
        X_tr_processed = preprocessor.fit_transform(X_tr, y_tr)
        X_val_processed = preprocessor.transform(X_val)
        
        # Train model
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X_tr_processed, y_tr)
        
        # Predict on validation
        oof_predictions[val_idx] = fold_model.predict(X_val_processed)
        
        models.append((preprocessor, fold_model))
    
    return models, oof_predictions