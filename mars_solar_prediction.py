"""
Mars Solar Panel Output Prediction System
==========================================
This script predicts daily solar panel output for a Martian colony during dust storms
using machine learning regression models.

Scientific Context:
- Mars receives ~43% of Earth's solar irradiance at perihelion
- Dust storms can reduce solar irradiance by 99% during global events
- Solar panel efficiency degrades due to dust accumulation and temperature variations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 1: GENERATE SYNTHETIC MARTIAN SOLAR DATA
# ============================================================================
# In a real scenario, this would be replaced with actual Mars rover/station data

def generate_mars_solar_data(n_samples=1000):
    """
    Generate synthetic Mars solar panel data with realistic physical constraints.
    
    Parameters:
    -----------
    n_samples : int
        Number of daily observations to generate
        
    Returns:
    --------
    pd.DataFrame : Dataset with features and target variable
    """
    print("Generating synthetic Martian solar data...")
    
    # Feature 1: Solar irradiance (W/m²)
    # Mars average: 590 W/m² (clear day), can drop to <10 W/m² during dust storms
    base_irradiance = np.random.normal(550, 100, n_samples)
    
    # Feature 2: Dust storm probability (0-1 scale)
    # Higher values indicate higher likelihood/intensity of dust storms
    dust_probability = np.random.beta(2, 5, n_samples)  # Skewed toward lower values
    
    # Feature 3: Panel efficiency (%)
    # Typical range: 15-25% for space-grade solar panels, reduced by dust
    base_efficiency = np.random.normal(20, 2, n_samples)
    
    # Feature 4: Martian sol (day number in mission)
    # Panel efficiency degrades over time due to dust accumulation
    sol_number = np.arange(n_samples)
    
    # Feature 5: Panel temperature (°C)
    # Mars surface temperature: -60°C average, affects efficiency
    panel_temp = np.random.normal(-45, 15, n_samples)
    
    # Feature 6: Atmospheric opacity (tau)
    # Measure of dust in atmosphere; normal ~0.5, storm >2.0
    atmospheric_opacity = 0.3 + dust_probability * 3.0 + np.random.normal(0, 0.2, n_samples)
    atmospheric_opacity = np.clip(atmospheric_opacity, 0.1, 5.0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'solar_irradiance': base_irradiance,
        'dust_storm_probability': dust_probability,
        'base_panel_efficiency': base_efficiency,
        'sol_number': sol_number,
        'panel_temperature': panel_temp,
        'atmospheric_opacity': atmospheric_opacity
    })
    
    # ========================================================================
    # CALCULATE TARGET: Daily Energy Output (kWh)
    # ========================================================================
    # Physics-based calculation with dust storm effects
    
    # Reduce irradiance based on atmospheric opacity
    effective_irradiance = data['solar_irradiance'] * np.exp(-atmospheric_opacity)
    
    # Panel efficiency decreases with dust storms and temperature
    dust_efficiency_factor = 1 - (data['dust_storm_probability'] * 0.6)
    temp_efficiency_factor = 1 - ((data['panel_temperature'] + 45) * 0.002)
    degradation_factor = 1 - (data['sol_number'] / n_samples * 0.15)  # 15% degradation over mission
    
    effective_efficiency = (data['base_panel_efficiency'] * 
                           dust_efficiency_factor * 
                           temp_efficiency_factor * 
                           degradation_factor)
    
    # Panel array size: 100 m² (typical for small Mars colony)
    panel_area = 100  # square meters
    
    # Mars sol is ~24.6 hours; assume ~12 hours of usable sunlight
    sunlight_hours = 12
    
    # Calculate daily energy output in kWh
    # Energy (kWh) = Power (kW) × Time (hours)
    # Power = Irradiance (W/m²) × Area (m²) × Efficiency (decimal)
    daily_output = (effective_irradiance * panel_area * 
                   (effective_efficiency / 100) * sunlight_hours) / 1000
    
    # Add realistic noise (measurement errors, weather variations)
    daily_output += np.random.normal(0, 5, n_samples)
    daily_output = np.clip(daily_output, 0, None)  # Energy cannot be negative
    
    data['daily_energy_output_kwh'] = daily_output
    
    print(f"✓ Generated {n_samples} samples")
    print(f"  Energy output range: {daily_output.min():.2f} - {daily_output.max():.2f} kWh")
    
    return data


# ============================================================================
# STEP 2: DATA PREPROCESSING AND EXPLORATION
# ============================================================================

def preprocess_data(data):
    """
    Perform data cleaning and exploratory analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame : Cleaned dataset
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Check for missing values
    missing = data.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    
    # Remove any duplicate rows
    initial_rows = len(data)
    data = data.drop_duplicates()
    print(f"Removed {initial_rows - len(data)} duplicate rows")
    
    # Statistical summary
    print(f"\nDataset shape: {data.shape}")
    print(f"\nStatistical Summary:")
    print(data.describe())
    
    # Check for outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    print(f"\nOutliers detected per feature:\n{outliers}")
    
    return data


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def engineer_features(data):
    """
    Create additional features to improve model performance.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed dataset
        
    Returns:
    --------
    pd.DataFrame : Dataset with engineered features
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # Interaction features
    data['irradiance_x_efficiency'] = (data['solar_irradiance'] * 
                                       data['base_panel_efficiency'])
    
    data['dust_opacity_interaction'] = (data['dust_storm_probability'] * 
                                       data['atmospheric_opacity'])
    
    # Polynomial features for non-linear relationships
    data['opacity_squared'] = data['atmospheric_opacity'] ** 2
    data['temp_squared'] = data['panel_temperature'] ** 2
    
    # Seasonal patterns (Mars year ≈ 687 Earth days)
    data['seasonal_component'] = np.sin(2 * np.pi * data['sol_number'] / 687)
    
    print(f"✓ Created {5} new engineered features")
    print(f"  Total features: {data.shape[1] - 1} (excluding target)")
    
    return data


# ============================================================================
# STEP 4: TRAIN-TEST SPLIT AND FEATURE SCALING
# ============================================================================

def prepare_train_test_data(data, test_size=0.2):
    """
    Split data into training and testing sets, then scale features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with all features
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple : X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("\n" + "="*70)
    print("DATA SPLITTING AND SCALING")
    print("="*70)
    
    # Separate features (X) and target (y)
    X = data.drop('daily_energy_output_kwh', axis=1)
    y = data['daily_energy_output_kwh']
    
    feature_names = X.columns.tolist()
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    print(f"Training set size: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set size: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    # Feature scaling using StandardScaler
    # This normalizes features to have mean=0 and std=1
    # Critical for distance-based algorithms, beneficial for tree-based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Features scaled using StandardScaler")
    print(f"  Mean: {scaler.mean_[:3]} ...")
    print(f"  Std: {scaler.scale_[:3]} ...")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================

def train_random_forest_model(X_train, y_train):
    """
    Train a Random Forest Regressor optimized for Mars solar prediction.
    
    Random Forest advantages:
    - Handles non-linear relationships well
    - Robust to outliers
    - Provides feature importance
    - Minimal hyperparameter tuning needed
    
    Parameters:
    -----------
    X_train : np.ndarray
        Scaled training features
    y_train : pd.Series
        Training target values
        
    Returns:
    --------
    RandomForestRegressor : Trained model
    """
    print("\n" + "="*70)
    print("MODEL TRAINING: Random Forest Regressor")
    print("="*70)
    
    # Initialize Random Forest with optimized hyperparameters
    model = RandomForestRegressor(
        n_estimators=200,        # Number of trees in the forest
        max_depth=20,            # Maximum depth of trees (prevents overfitting)
        min_samples_split=5,     # Minimum samples required to split a node
        min_samples_leaf=2,      # Minimum samples required at leaf node
        max_features='sqrt',     # Number of features to consider for best split
        random_state=42,         # Reproducibility
        n_jobs=-1,               # Use all CPU cores
        verbose=0
    )
    
    print("Hyperparameters:")
    print(f"  - Number of trees: {model.n_estimators}")
    print(f"  - Max depth: {model.max_depth}")
    print(f"  - Min samples split: {model.min_samples_split}")
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Perform cross-validation to assess model stability
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=5, scoring='r2', n_jobs=-1)
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model


# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Parameters:
    -----------
    model : trained model
        The trained Random Forest model
    X_train, X_test : np.ndarray
        Scaled feature sets
    y_train, y_test : pd.Series
        Target values
    feature_names : list
        Names of features for importance ranking
        
    Returns:
    --------
    tuple : (y_pred_train, y_pred_test, metrics_dict)
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate regression metrics
    # 1. Mean Squared Error (MSE): Average of squared differences
    #    Lower is better; sensitive to outliers
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    # 2. Root Mean Squared Error (RMSE): Square root of MSE
    #    Same units as target variable
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    
    # 3. Mean Absolute Error (MAE): Average of absolute differences
    #    More robust to outliers than MSE
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # 4. R² Score (Coefficient of Determination)
    #    Proportion of variance explained by model; 1.0 is perfect
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Print results
    print("\nTRAINING SET PERFORMANCE:")
    print(f"  R² Score:  {r2_train:.4f}")
    print(f"  RMSE:      {rmse_train:.4f} kWh")
    print(f"  MAE:       {mae_train:.4f} kWh")
    print(f"  MSE:       {mse_train:.4f}")
    
    print("\nTEST SET PERFORMANCE:")
    print(f"  R² Score:  {r2_test:.4f}")
    print(f"  RMSE:      {rmse_test:.4f} kWh")
    print(f"  MAE:       {mae_test:.4f} kWh")
    print(f"  MSE:       {mse_test:.4f}")
    
    # Assess overfitting
    r2_diff = r2_train - r2_test
    print(f"\nOverfitting Assessment:")
    print(f"  R² difference (train - test): {r2_diff:.4f}")
    if r2_diff < 0.05:
        print("  ✓ Model generalizes well (minimal overfitting)")
    elif r2_diff < 0.15:
        print("  ⚠ Slight overfitting detected")
    else:
        print("  ✗ Significant overfitting - consider regularization")
    
    # Feature importance analysis
    print("\nFEATURE IMPORTANCE (Top 10):")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Store metrics
    metrics = {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'mse_train': mse_train,
        'mse_test': mse_test
    }
    
    return y_pred_train, y_pred_test, metrics


# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

def plot_results(y_train, y_test, y_pred_train, y_pred_test, metrics):
    """
    Create comprehensive visualization of model performance.
    
    Parameters:
    -----------
    y_train, y_test : pd.Series
        Actual target values
    y_pred_train, y_pred_test : np.ndarray
        Predicted values
    metrics : dict
        Evaluation metrics
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mars Solar Panel Output Prediction - Model Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # -------------------------------------------------------------------------
    # Plot 1: Predicted vs Actual (Training Set)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.scatter(y_train, y_pred_train, alpha=0.5, s=20, color='blue', edgecolors='k', linewidth=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Output (kWh)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Predicted Output (kWh)', fontsize=10, fontweight='bold')
    ax1.set_title(f'Training Set: R² = {metrics["r2_train"]:.4f}', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Predicted vs Actual (Test Set)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred_test, alpha=0.6, s=25, color='green', edgecolors='k', linewidth=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Output (kWh)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Predicted Output (kWh)', fontsize=10, fontweight='bold')
    ax2.set_title(f'Test Set: R² = {metrics["r2_test"]:.4f}', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Residual Plot (Test Set)
    # -------------------------------------------------------------------------
    ax3 = axes[0, 2]
    residuals = y_test - y_pred_test
    ax3.scatter(y_pred_test, residuals, alpha=0.6, s=25, color='purple', edgecolors='k', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Output (kWh)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Residuals (kWh)', fontsize=10, fontweight='bold')
    ax3.set_title('Residual Plot (Test Set)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Error Distribution (Test Set)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 0]
    ax4.hist(residuals, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Prediction Error (kWh)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title(f'Error Distribution (MAE: {metrics["mae_test"]:.2f} kWh)', 
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 5: Prediction Timeline Comparison
    # -------------------------------------------------------------------------
    ax5 = axes[1, 1]
    test_indices = range(len(y_test))
    ax5.plot(test_indices[:200], y_test.values[:200], 'o-', label='Actual', 
             markersize=4, linewidth=1, alpha=0.7)
    ax5.plot(test_indices[:200], y_pred_test[:200], 's-', label='Predicted', 
             markersize=4, linewidth=1, alpha=0.7)
    ax5.set_xlabel('Sample Index', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Energy Output (kWh)', fontsize=10, fontweight='bold')
    ax5.set_title('Prediction Timeline (First 200 Test Samples)', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 6: Metrics Comparison Bar Chart
    # -------------------------------------------------------------------------
    ax6 = axes[1, 2]
    metrics_names = ['R² Score', 'RMSE (kWh)', 'MAE (kWh)']
    train_values = [metrics['r2_train'], metrics['rmse_train'], metrics['mae_train']]
    test_values = [metrics['r2_test'], metrics['rmse_test'], metrics['mae_test']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, train_values, width, label='Training', 
                    color='skyblue', edgecolor='black')
    bars2 = ax6.bar(x + width/2, test_values, width, label='Testing', 
                    color='lightcoral', edgecolor='black')
    
    ax6.set_ylabel('Metric Value', fontsize=10, fontweight='bold')
    ax6.set_title('Performance Metrics Comparison', fontsize=11, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_names)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'mars_solar_prediction_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    plt.show()


# ============================================================================
# STEP 8: MODEL PERSISTENCE
# ============================================================================

def save_model_artifacts(model, scaler, feature_names, metrics):
    """
    Save trained model and associated artifacts for future use.
    
    Parameters:
    -----------
    model : trained model
        The trained Random Forest model
    scaler : StandardScaler
        Fitted scaler for feature transformation
    feature_names : list
        Names of features
    metrics : dict
        Model performance metrics
    """
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    # Save the trained model
    model_path = 'mars_solar_model.joblib'
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save the scaler
    scaler_path = 'mars_solar_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature names and metadata
    metadata = {
        'feature_names': feature_names,
        'metrics': metrics,
        'model_type': 'RandomForestRegressor',
        'n_features': len(feature_names)
    }
    metadata_path = 'mars_solar_metadata.joblib'
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\nModel artifacts ready for deployment!")
    print("To load the model later:")
    print("  model = joblib.load('mars_solar_model.joblib')")
    print("  scaler = joblib.load('mars_solar_scaler.joblib')")


# ============================================================================
# STEP 9: PREDICTION FUNCTION FOR NEW DATA
# ============================================================================

def predict_solar_output(model, scaler, feature_names, new_data):
    """
    Make predictions on new Martian solar data.
    
    Parameters:
    -----------
    model : trained model
        Loaded model
    scaler : StandardScaler
        Loaded scaler
    feature_names : list
        Expected feature names
    new_data : pd.DataFrame
        New input data with same features
        
    Returns:
    --------
    np.ndarray : Predicted solar output values
    """
    # Ensure features match training data
    if not all(feat in new_data.columns for feat in feature_names):
        missing = [f for f in feature_names if f not in new_data.columns]
        raise ValueError(f"Missing features: {missing}")
    
    # Select and order features
    X_new = new_data[feature_names]
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Generate predictions
    predictions = model.predict(X_new_scaled)
    
    return predictions


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution function orchestrating the entire ML pipeline.
    """
    print("\n" + "="*70)
    print("MARS COLONY SOLAR PANEL OUTPUT PREDICTION SYSTEM")
    print("="*70)
    print("Objective: Predict daily solar energy generation during dust storms")
    print("Model: Random Forest Regression")
    print("="*70)
    
    # Step 1: Generate data
    data = generate_mars_solar_data(n_samples=1000)
    
    # Step 2: Preprocess
    data = preprocess_data(data)
    
    # Step 3: Feature engineering
    data = engineer_features(data)
    
    # Step 4: Prepare train/test data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_data(data)
    
    # Step 5: Train model
    model = train_random_forest_model(X_train, y_train)
    
    # Step 6: Evaluate model
    y_pred_train, y_pred_test, metrics = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_names
    )
    
    # Step 7: Visualize results
    plot_results(y_train, y_test, y_pred_train, y_pred_test, metrics)
    
    # Step 8: Save model
    save_model_artifacts(model, scaler, feature_names, metrics)
    
    # ========================================================================
    # DEMONSTRATION: Making predictions with saved model
    # ========================================================================
    print("\n" + "="*70)
    print("DEMONSTRATION: LOADING MODEL AND MAKING PREDICTIONS")
    print("="*70)
    
    # Simulate loading the model
    loaded_model = joblib.load('mars_solar_model.joblib')
    loaded_scaler = joblib.load('mars_solar_scaler.joblib')
    loaded_metadata = joblib.load('mars_solar_metadata.joblib')
    
    print("✓ Model loaded successfully")
    print(f"  Model type: {loaded_metadata['model_type']}")
    print(f"  Number of features: {loaded_metadata['n_features']}")
    print(f"  Test R² score: {loaded_metadata['metrics']['r2_test']:.4f}")
    
    # Create sample prediction data
    sample_data = pd.DataFrame({
        'solar_irradiance': [500],
        'dust_storm_probability': [0.7],
        'base_panel_efficiency': [18],
        'sol_number': [500],
        'panel_temperature': [-40],
        'atmospheric_opacity': [2.1]
    })
    
    # Engineer features for sample (same as training)
    sample_data['irradiance_x_efficiency'] = (sample_data['solar_irradiance'] * 
                                               sample_data['base_panel_efficiency'])
    sample_data['dust_opacity_interaction'] = (sample_data['dust_storm_probability'] * 
                                               sample_data['atmospheric_opacity'])
    sample_data['opacity_squared'] = sample_data['atmospheric_opacity'] ** 2
    sample_data['temp_squared'] = sample_data['panel_temperature'] ** 2
    sample_data['seasonal_component'] = np.sin(2 * np.pi * sample_data['sol_number'] / 687)
    
    # Make prediction
    prediction = predict_solar_output(loaded_model, loaded_scaler, 
                                     loaded_metadata['feature_names'], sample_data)
    
    print(f"\nSample Prediction (moderate dust storm):")
    print(f"  Input conditions:")
    print(f"    - Solar irradiance: 500 W/m²")
    print(f"    - Dust storm probability: 0.7 (70%)")
    print(f"    - Panel efficiency: 18%")
    print(f"    - Atmospheric opacity: 2.1 (high dust)")
    print(f"  Predicted daily output: {prediction[0]:.2f} kWh")
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print("\nFiles generated:")
    print("  1. mars_solar_model.joblib (trained model)")
    print("  2. mars_solar_scaler.joblib (feature scaler)")
    print("  3. mars_solar_metadata.joblib (model metadata)")
    print("  4. mars_solar_prediction_results.png (visualizations)")
    print("\nReady for deployment in Mars colony operations!")


# ============================================================================
# EXECUTE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()
