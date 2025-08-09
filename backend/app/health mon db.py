# async Mongo client wrapper
import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/healthguardian")
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()
users = db.users
logs = db.activity_logs
recommendations = db.recommendations

