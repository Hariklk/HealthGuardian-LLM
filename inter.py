from pymongo import MongoClient
import datetime


client = MongoClient("mongodb://localhost:27017/")
db = client["health_routine_db"]

user_data = {
    "name": "John Doe",
    "email": "johndoe@example.com",
    "age": 30,
    "gender": "Male",
    "height_cm": 175,
    "weight_kg": 70,
    "medical_conditions": ["Diabetes", "Hypertension"],
    "goals": {
        "weight_loss": True,
        "muscle_gain": False,
        "improve_sleep": True
    },
    "created_at": datetime.datetime.utcnow()
}
db.users.insert_one(user_data)

print("✅ User inserted successfully!")
