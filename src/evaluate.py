import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from preprocess import prepare_data

def main():
    print("Loading test data and trained model...")
    X_test, y_test = prepare_data("data/test.csv")
    model = joblib.load("models/qsar_model.pkl")

    print("Running predictions...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n--- Evaluation Results ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")

if __name__ == "__main__":
    main()
