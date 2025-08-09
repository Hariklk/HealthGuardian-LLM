fetch("http://localhost:5000/api/health-data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...userData })
})
