import React, {useState} from "react";
import { createUser, recommend } from "./api";

export default function App(){
  const [email, setEmail] = useState("");
  const [uid, setUid] = useState("");
  const [sleep, setSleep] = useState(7);
  const [activity, setActivity] = useState("exercise");
  const [result, setResult] = useState(null);

  const handleCreate = async ()=>{
    const res = await createUser({name:"Demo", email, birth_year:1990, health_constraints:{heart_condition:false}});
    setUid(res.id);
  };

  const handleRecommend = async ()=>{
    if(!uid){ alert("Create user first"); return; }
    const res = await recommend({user_id: uid, activity, scheduled_duration_min:30, sleep_hours: parseFloat(sleep)});
    setResult(res);
  };

  return (<div style={{padding:20}}>
    <h1>HealthGuardian Demo</h1>
    <div>
      <input placeholder="email" value={email} onChange={e=>setEmail(e.target.value)} />
      <button onClick={handleCreate}>Create User</button>
    </div>

    <div style={{marginTop:10}}>
      <label>Activity: <input value={activity} onChange={e=>setActivity(e.target.value)} /></label>
      <label>Sleep hrs: <input type="number" value={sleep} onChange={e=>setSleep(e.target.value)} /></label>
      <button onClick={handleRecommend}>Get Recommendation</button>
    </div>

    {result && <div style={{marginTop:20}}>
      <h3>Recommendation</h3>
      <p>Best time: {result.best_time} (minutes since midnight)</p>
      <p>Predicted completion probability: {(result.predicted_completion_prob||0).toFixed(2)}</p>
      <p>Safe: {String(result.safe)}</p>
      <p>Explanation: {result.explanation}</p>
    </div>}
  </div>);
}
