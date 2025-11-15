import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def import_data(filename: str=r'Vanderbilt Data.xlsx') -> pd.DataFrame:
    df = pd.read_excel(filename, sheet_name='Sheet1')
    # Fix whitespace
    df.columns = df.columns.str.replace(" - ", "_")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # date features
    df['SurgDate'] = pd.to_datetime(df['SurgDate'])
    df['month'] = df['SurgDate'].dt.month
    df['weekofyear'] = df['SurgDate'].dt.isocalendar().week.astype(int)
    df['day'] = df['SurgDate'].dt.day
    # categorize day of week
    df['DOW'] = df['DOW'].astype('category')
    # add delta features for better predictions -- shows change over time more directly
    df['d7_14'] = df['T_7'] - df['T_14']
    df['d3_7']  = df['T_3'] - df['T_7']
    df['d1_3']  = df['T_1'] - df['T_3']
    df['d1_7']  = df['T_1'] - df['T_7']
    df['d1_14'] = df['T_1'] - df['T_14']
    # weekday average to account for seasonality / weekday patterns
    df['dow_avg'] = df.groupby('DOW')['Actual'].transform('mean')
    df['month_avg'] = df.groupby('month')['Actual'].transform('mean')

    return df


def train_lgbm_model(df, feature_cols, target_col='Actual', n_splits=2):
    X = df[feature_cols]
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    maes, rmses, r2s = [], [], []

    for train_idx, test_idx in tscv.split(df):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=8,
            min_data_in_leaf=5,
            subsample=0.9,
            colsample_bytree=0.8,
            objective='regression_l1',
            random_state=42
        )

        model.fit(X_train, y_train, categorical_feature=['DOW'])

        preds = model.predict(X_test)
        # Compute metrics
        maes.append(mean_absolute_error(y, preds))
        rmses.append(root_mean_squared_error(y, preds))
        r2s.append(r2_score(y, preds))

    # Train final model on full dataset
    final_model = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=8,
        min_data_in_leaf=5,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='regression_l1',
        random_state=42
    )
    final_model.fit(X, y, categorical_feature=['DOW'])

    stats = {
        'MAE': np.mean(maes),
        'RMSE': np.mean(rmses),
        'R2': np.mean(r2s)
    }

    return final_model, stats, X, y


def generate_model_summary(df: pd.DataFrame, model, feature_cols):
    """
    Generates a single-page model effectiveness summary:
    - Top 10 feature importances
    - Residuals plot
    - Predicted vs Actual scatter
    - Error distribution histogram
    - Time series of predictions vs actual
    - MAE, RMSE, R²
    """
    # Make predictions
    X = df[feature_cols]
    y = df['Actual']
    preds = model.predict(X)

    # Compute metrics
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    # Add predictions and residuals to dataframe
    df_plot = df.copy()
    df_plot['Predicted'] = preds
    df_plot['Residual'] = df_plot['Actual'] - df_plot['Predicted']

    # Sort by date for time series plot
    if not pd.api.types.is_datetime64_any_dtype(df_plot['SurgDate']):
        df_plot['SurgDate'] = pd.to_datetime(df_plot['SurgDate'])
    df_plot = df_plot.sort_values('SurgDate')

    # Set up figure
    plt.figure(figsize=(15, 12))
    
    # Display summary stats as the figure title
    plt.suptitle(f"Model Effectiveness Summary\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}",
                fontsize=16, y=1.02)  # y > 1 moves it above the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave more space for suptitle

    # 1. Top 10 Feature Importances
    plt.subplot(3, 2, 1)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=False).head(10)
    sns.barplot(x=top10.values, y=top10.index, palette="viridis")
    plt.title('Top 10 Feature Importances')

    # 2. Residuals vs Predicted
    plt.subplot(3, 2, 2)
    sns.scatterplot(x=df_plot['Predicted'], y=df_plot['Residual'])
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Residuals vs Predicted')

    # 3. Predicted vs Actual Scatter
    plt.subplot(3, 2, 3)
    sns.scatterplot(x=df_plot['Actual'], y=df_plot['Predicted'])
    plt.plot([df_plot['Actual'].min(), df_plot['Actual'].max()],
             [df_plot['Actual'].min(), df_plot['Actual'].max()],
             color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')

    # 4. Error Distribution Histogram
    plt.subplot(3, 2, 4)
    sns.histplot(df_plot['Residual'], kde=True, bins=20, color='skyblue')
    plt.xlabel('Residual')
    plt.title('Error Distribution')

    # 5. Time Series of Predicted vs Actual
    plt.subplot(3, 1, 3)
    plt.plot(df_plot['SurgDate'], df_plot['Actual'], marker='o', linestyle='-', label='Actual')
    plt.plot(df_plot['SurgDate'], df_plot['Predicted'], marker='x', linestyle='--', label='Predicted')
    plt.xlabel('Surgery Date')
    plt.ylabel('OR Cases')
    plt.title('Time Series: Predicted vs Actual')
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    df = import_data()
    df = add_features(df)
    # Define features
    feature_cols = [
        'DOW','month','weekofyear','day',
        'T_28','T_21','T_14','T_13','T_12','T_11',
        'T_10','T_9','T_8','T_7','T_6','T_5',
        'T_4','T_3','T_2','T_1',
        'd7_14','d3_7','d1_3','d1_7','d1_14',
        'dow_avg','month_avg'
    ]

    # Train model
    model, stats, X, y = train_lgbm_model(df, feature_cols)

    # Generate report
    generate_model_summary(df, model, feature_cols)