from fastapi import FastAPI, HTTPException
from .db import users, logs, recommendations
from .schemas import UserCreate, RecommendRequest
from bson import ObjectId
from .recommend import generate_recommendation
import asyncio

app = FastAPI(title="HealthGuardian-LLM")

@app.post("/users")
async def create_user(u: UserCreate):
    doc = u.dict()
    res = await users.insert_one(doc)
    return {"id": str(res.inserted_id)}

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    doc = await users.find_one({"_id": ObjectId(user_id)})
    if not doc:
        raise HTTPException(404, "User not found")
    doc["id"] = str(doc["_id"])
    return doc

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    user_doc = await users.find_one({"_id": ObjectId(req.user_id)})
    if not user_doc:
        raise HTTPException(404, "User not found")
    rec = await generate_recommendation(user_doc, req)
    # store recap
    await recommendations.insert_one({**rec, "user_id": req.user_id})
    return rec
