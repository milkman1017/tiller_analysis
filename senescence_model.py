import matplotlib
matplotlib.use('Agg')  # Use the non-graphical Agg backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Custom callback for F1 score
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred_prob = self.model.predict(X_val).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        f1 = f1_score(y_val, y_pred)
        print(f"\nEpoch {epoch + 1}: F1 Score = {f1:.4f}")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(data, numeric_features, categorical_feature):
    """
    Dynamically preprocess data by scaling numeric features and one-hot encoding categorical features.
    """
    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [categorical_feature])
        ],
        remainder='drop'
    )

    # Transform the data
    transformed_data = preprocessor.fit_transform(data)

    # Get feature names dynamically
    numeric_feature_names = numeric_features
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out([categorical_feature])
    all_feature_names = numeric_feature_names + list(categorical_feature_names)

    # Convert the transformed data to a DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)

    return transformed_df, preprocessor

from tensorflow.keras.layers import LSTM, TimeDistributed, Flatten

def create_sliding_window_data(data, transformed_data, features, target, time_steps, group_by):
    """
    Create sliding window data for LSTM using original data for grouping and transformed data for features.
    Args:
        data (DataFrame): Original data for grouping purposes.
        transformed_data (DataFrame): Transformed data for features.
        features (list): List of feature columns.
        target (str): Target column.
        time_steps (int): Number of time steps in the sliding window.
        group_by (list): Columns to group data (e.g., ['year', 'Site']).
    Returns:
        tuple: (X, y) arrays for the LSTM model.
    """
    X, y = [], []
    grouped_data = data.groupby(group_by)

    for _, group in grouped_data:
        # Get the indices of the group in the original data
        group_indices = group.index
        # Use these indices to slice the transformed data
        group_features = transformed_data.iloc[group_indices].values
        group_target = group[target].values

        # Create sliding windows
        for i in range(len(group) - time_steps + 1):
            X.append(group_features[i:i + time_steps])
            y.append(group_target[i + time_steps - 1])  # Predicting the last step

    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """
    Build an LSTM model for time-series data.
    
    Args:
        input_shape (tuple): Shape of the input data (time_steps, num_features).
    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def optimize_threshold(y_true, y_pred_prob):
    """
    Find the optimal threshold for F1 score.
    """
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_true, (y_pred_prob >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {max(f1_scores):.4f}")
    return best_threshold


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    Args:
        model (Sequential): Keras model.
        X_test (array): Test features.
        y_test (array): Test labels.
    """
    # Predict probabilities and binarize
    y_pred_prob = model.predict(X_test).flatten()
    best_threshold = optimize_threshold(y_test, y_pred_prob)
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {roc_auc:.2f}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()


def shap_interpretation_lstm(model, X_train, X_test, preprocessor, time_steps):
    """
    Use SHAP to interpret model predictions for LSTM.
    """
    # Define a wrapper to reshape input for SHAP
    def model_predict(data):
        # Reshape the data from 2D back to 3D for LSTM input
        reshaped_data = data.reshape(-1, time_steps, X_train.shape[2])
        return model.predict(reshaped_data).flatten()  # Return 1D array for SHAP compatibility

    # Select a subset of the training data for SHAP (for efficiency)
    explainer_data = X_train[:100].reshape(-1, time_steps * X_train.shape[2])  # Flatten for SHAP
    X_test_flat = X_test.reshape(-1, time_steps * X_test.shape[2])  # Flatten for SHAP

    # Create SHAP KernelExplainer
    explainer = shap.KernelExplainer(model_predict, explainer_data)

    # Compute SHAP values for test samples
    shap_values = explainer.shap_values(X_test_flat[:10])

    # Ensure feature names match SHAP values dimensions
    feature_names = preprocessor.get_feature_names_out()
    expanded_feature_names = [f"{name}_t{i}" for i in range(time_steps) for name in feature_names]

    # Check alignment between SHAP values and feature names
    shap_values_flat = np.array(shap_values)  # SHAP returns a list; convert to array for consistency
    if shap_values_flat.shape[1] != len(expanded_feature_names):
        raise ValueError(
            f"Feature names dimension {len(expanded_feature_names)} "
            f"does not match SHAP values dimension {shap_values_flat.shape[1]}."
        )

    # SHAP Summary Plot
    shap.summary_plot(shap_values_flat, X_test_flat[:10], feature_names=expanded_feature_names)
    plt.savefig('shap_summary_plot.png')
    print("SHAP summary plot saved as 'shap_summary_plot.png'.")




def main():
    file_path = "ArcticScience25-LGLdata_with_weather_data.csv"  # Replace with your CSV file path
    data = pd.read_csv(file_path)

    # Define numeric features and target
    numeric_features = ['prcp (mm/day)', 'srad (W/m^2)', 'swe (kg/m^2)', 'tmax (deg c)', 'tmin (deg c)', 'vp (Pa)']
    categorical_feature = 'Site'
    target = 'senescence_triggered'

    # Preprocess the data
    transformed_data, preprocessor = preprocess_data(data, numeric_features, categorical_feature)

    # Dynamically handle site categories in sliding window
    time_steps = 7
    group_by = ['year', 'Site']  # Group by year and Site (as in the original data)
    X, y = create_sliding_window_data(data, transformed_data, transformed_data.columns, target, time_steps, group_by)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights}")

    # Build and train the LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    f1_callback = F1ScoreCallback(validation_data=(X_test, y_test))

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Metric to monitor
        factor=0.5,          # Factor by which to reduce the learning rate
        patience=5,         # Number of epochs with no improvement before reducing
        min_lr=1e-6,         # Minimum learning rate
        verbose=1            # Print updates to the console
    )

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=1028,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[f1_callback, reduce_lr]
    )

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Interpret the model using SHAP
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    shap_interpretation_lstm(model, X_train, X_test, preprocessor, time_steps)


if __name__ == "__main__":
    main()
