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

np.random.seed(42)

def generate_mars_solar_data(n_samples=1000):
    print("Generating synthetic Martian solar data...")
    
    base_irradiance = np.random.normal(550, 100, n_samples)
    dust_probability = np.random.beta(2, 5, n_samples)
    base_efficiency = np.random.normal(20, 2, n_samples)
    sol_number = np.arange(n_samples)
    panel_temp = np.random.normal(-45, 15, n_samples)
    atmospheric_opacity = 0.3 + dust_probability * 3.0 + np.random.normal(0, 0.2, n_samples)
    atmospheric_opacity = np.clip(atmospheric_opacity, 0.1, 5.0)
    
    data = pd.DataFrame({
        'solar_irradiance': base_irradiance,
        'dust_storm_probability': dust_probability,
        'base_panel_efficiency': base_efficiency,
        'sol_number': sol_number,
        'panel_temperature': panel_temp,
        'atmospheric_opacity': atmospheric_opacity
    })
    
    effective_irradiance = data['solar_irradiance'] * np.exp(-atmospheric_opacity)
    
    dust_efficiency_factor = 1 - (data['dust_storm_probability'] * 0.6)
    temp_efficiency_factor = 1 - ((data['panel_temperature'] + 45) * 0.002)
    degradation_factor = 1 - (data['sol_number'] / n_samples * 0.15)
    
    effective_efficiency = (data['base_panel_efficiency'] * 
                           dust_efficiency_factor * 
                           temp_efficiency_factor * 
                           degradation_factor)
    
    panel_area = 100
    sunlight_hours = 12
    
    daily_output = (effective_irradiance * panel_area * 
                   (effective_efficiency / 100) * sunlight_hours) / 1000
    
    daily_output += np.random.normal(0, 5, n_samples)
    daily_output = np.clip(daily_output, 0, None)
    
    data['daily_energy_output_kwh'] = daily_output
    
    print(f"✓ Generated {n_samples} samples")
    print(f"  Energy output range: {daily_output.min():.2f} - {daily_output.max():.2f} kWh")
    
    return data


def preprocess_data(data):
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    missing = data.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    
    initial_rows = len(data)
    data = data.drop_duplicates()
    print(f"Removed {initial_rows - len(data)} duplicate rows")
    
    print(f"\nDataset shape: {data.shape}")
    print(f"\nStatistical Summary:")
    print(data.describe())
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    print(f"\nOutliers detected per feature:\n{outliers}")
    
    return data


def engineer_features(data):
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    data['irradiance_x_efficiency'] = (data['solar_irradiance'] * 
                                       data['base_panel_efficiency'])
    
    data['dust_opacity_interaction'] = (data['dust_storm_probability'] * 
                                       data['atmospheric_opacity'])
    
    data['opacity_squared'] = data['atmospheric_opacity'] ** 2
    data['temp_squared'] = data['panel_temperature'] ** 2
    
    data['seasonal_component'] = np.sin(2 * np.pi * data['sol_number'] / 687)
    
    print(f"✓ Created {5} new engineered features")
    print(f"  Total features: {data.shape[1] - 1} (excluding target)")
    
    return data


def prepare_train_test_data(data, test_size=0.2):
    print("\n" + "="*70)
    print("DATA SPLITTING AND SCALING")
    print("="*70)
    
    X = data.drop('daily_energy_output_kwh', axis=1)
    y = data['daily_energy_output_kwh']
    
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    print(f"Training set size: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set size: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Features scaled using StandardScaler")
    print(f"  Mean: {scaler.mean_[:3]} ...")
    print(f"  Std: {scaler.scale_[:3]} ...")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_random_forest_model(X_train, y_train):
    print("\n" + "="*70)
    print("MODEL TRAINING: Random Forest Regressor")
    print("="*70)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("Hyperparameters:")
    print(f"  - Number of trees: {model.n_estimators}")
    print(f"  - Max depth: {model.max_depth}")
    print(f"  - Min samples split: {model.min_samples_split}")
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=5, scoring='r2', n_jobs=-1)
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
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
    
    r2_diff = r2_train - r2_test
    print(f"\nOverfitting Assessment:")
    print(f"  R² difference (train - test): {r2_diff:.4f}")
    if r2_diff < 0.05:
        print("  ✓ Model generalizes well (minimal overfitting)")
    elif r2_diff < 0.15:
        print("  ⚠ Slight overfitting detected")
    else:
        print("  ✗ Significant overfitting - consider regularization")
    
    print("\nFEATURE IMPORTANCE (Top 10):")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
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


def plot_results(y_train, y_test, y_pred_train, y_pred_test, metrics):
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mars Solar Panel Output Prediction - Model Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
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
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = 'mars_solar_prediction_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    plt.show()


def save_model_artifacts(model, scaler, feature_names, metrics):
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    model_path = 'mars_solar_model.joblib'
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    scaler_path = 'mars_solar_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
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


def predict_solar_output(model, scaler, feature_names, new_data):
    if not all(feat in new_data.columns for feat in feature_names):
        missing = [f for f in feature_names if f not in new_data.columns]
        raise ValueError(f"Missing features: {missing}")
    
    X_new = new_data[feature_names]
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    
    return predictions


def main():
    print("\n" + "="*70)
    print("MARS COLONY SOLAR PANEL OUTPUT PREDICTION SYSTEM")
    print("="*70)
    print("Objective: Predict daily solar energy generation during dust storms")
    print("Model: Random Forest Regression")
    print("="*70)
    
    data = generate_mars_solar_data(n_samples=1000)
    data = preprocess_data(data)
    data = engineer_features(data)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_data(data)
    model = train_random_forest_model(X_train, y_train)
    y_pred_train, y_pred_test, metrics = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_names
    )
    plot_results(y_train, y_test, y_pred_train, y_pred_test, metrics)
    save_model_artifacts(model, scaler, feature_names, metrics)
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


if __name__ == "__main__":
    main()
