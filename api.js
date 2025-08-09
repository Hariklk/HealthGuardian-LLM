const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000"

export async function postJSON(path, body){
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  return r.json();
}

export function createUser(u){
  return postJSON("/users", u);
}
export function recommend(body){
  return postJSON("/recommend", body);
}
