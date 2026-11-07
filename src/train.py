import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from preprocess import prepare_data

def main():
    print("Loading and processing training data...")
    # Make sure you place a sample train.csv in the data/ folder
    X_train, y_train = prepare_data("data/train.csv")

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/qsar_model.pkl")
    print("Model successfully trained and saved to models/qsar_model.pkl")

if __name__ == "__main__":
    main()
