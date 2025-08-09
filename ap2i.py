from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os

app = Flask(__name__)
CORS(app)  # allow frontend requests

# MongoDB connection (replace with your URI)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["health_ai"]
users_collection = db["health_data"]

@app.route("/api/health-data", methods=["POST"])
def save_health_data():
    try:
        data = request.json

        # Validate required fields
        required = ["name", "age", "gender", "height", "weight", "activity", "diet"]
        if not all(field in data and data[field] for field in required):
            return jsonify({"message": "Missing required fields"}), 400

        # Save to MongoDB
        users_collection.insert_one(data)

        return jsonify({"message": "Health data saved successfully"}), 201

    except Exception as e:
        return jsonify({"message": "Error saving data", "error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Health AI Backend Running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
