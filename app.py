import pandas as pd
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Define the path where the model and features will be saved/loaded
MODEL_PATH = 'breast_cancer_model.pkl'
FEATURES_PATH = 'model_features.pkl'

# Define the features that the model expects
# This list should match the features used during model training in your notebook
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']

def load_model_and_features():
    """
    Loads the trained RandomForestClassifier model and the list of features.
    This function expects the model to be pre-trained and saved.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        print("Loading existing model and features...")
        try:
            model = joblib.load(MODEL_PATH)
            loaded_features = joblib.load(FEATURES_PATH)

            # Validate that the loaded features match the expected features
            if loaded_features != features:
                print("Error: Loaded model's features do not match the application's expected features.")
                print("Please ensure your Jupyter Notebook trains the model with the correct features and saves them.")
                raise ValueError("Feature mismatch: Model needs retraining with correct features.")
            print("Model and features loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model or features: {e}")
            print("Please ensure the model was trained and saved correctly by your Jupyter Notebook.")
            raise RuntimeError("Failed to load pre-trained model. Please train the model first.")
    else:
        print("Model files not found.")
        print("Please run your Jupyter Notebook to train the model and save it to 'breast_cancer_model.pkl' and 'model_features.pkl'.")
        raise FileNotFoundError("Pre-trained model files are missing. Please train the model first.")

# Load the model when the Flask application starts
try:
    model = load_model_and_features()
except (FileNotFoundError, RuntimeError, ValueError) as e:
    # If model loading fails, the app won't start, or will show an error on access
    # For a real application, you might want a more graceful error page
    print(f"Application startup failed due to model loading error: {e}")
    model = None # Set model to None to prevent further errors if app continues to run

@app.route('/')
def home():
    """
    Renders the home page with the input form.
    """
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded. Please train the model first.", confidence_text="")
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    """
    if model is None:
        return render_template('index.html', prediction_text="Error: Model is not available for prediction.", confidence_text="")

    try:
        # Get form data from the HTML form
        data = [float(request.form[feature]) for feature in features]

        # Create a Pandas DataFrame from the input data
        new_df = pd.DataFrame([data], columns=features)

        # Make prediction using the loaded model
        single_prediction = model.predict(new_df)[0]
        # Get the probability of the positive class (Malignant, which is class 1)
        proba = model.predict_proba(new_df)[:, 1][0]

        # Determine the output message based on the prediction
        if single_prediction == 1:
            output = "The patient has breast cancer (Malignant)."
            confidence = f"Confidence: {proba * 100:.2f}%"
        else:
            output = "The patient does not have breast cancer (Benign)."
            # For benign predictions (class 0), confidence is 1 - probability of class 1
            confidence = f"Confidence: {(1 - proba) * 100:.2f}%"

        # Render the HTML template with the prediction results
        return render_template('index.html', prediction_text=output, confidence_text=confidence)

    except ValueError:
        # Handle cases where input values are not valid numbers
        return render_template('index.html', prediction_text="Invalid input. Please enter numerical values for all fields.", confidence_text="")
    except Exception as e:
        # Catch any other unexpected errors during prediction
        return render_template('index.html', prediction_text=f"An unexpected error occurred: {e}", confidence_text="")

if __name__ == '__main__':
    # To run the Flask application, use 'flask run' in your terminal
    # Make sure to set FLASK_APP=app.py and FLASK_DEBUG=1 (for development)
    # Example:
    # export FLASK_APP=app.py
    # export FLASK_DEBUG=1
    # flask run
    app.run(debug=True)