from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.custom_data import CustomData

app = Flask(__name__)

# Initialize prediction pipeline once (GOOD practice)
predictor = PredictPipeline()


@app.route("/")
def home():
    return "Employee Churn Prediction API"


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint for employee churn prediction.
    Expects JSON input with employee features.
    """

    try:
        data = request.json

        # Convert raw JSON into CustomData object
        custom_data = CustomData(
            satisfaction_level=data["satisfaction_level"],
            last_evaluation=data["last_evaluation"],
            number_project=data["number_project"],
            average_montly_hours=data["average_montly_hours"],
            time_spend_company=data["time_spend_company"],
            work_accident=data["work_accident"],
            promotion_last_5years=data["promotion_last_5years"],
            salary_low=data["salary_low"],
            salary_medium=data["salary_medium"]
        )

        # Convert to DataFrame
        df = custom_data.get_data_as_dataframe()

        # Make prediction
        prediction = predictor.predict(df)[0]

        return jsonify({
            "prediction": int(prediction),
            "label": "Likely to leave" if prediction == 1 else "Likely to stay"
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
