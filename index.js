const express = require("express");
const { exec } = require("child_process");
const app = express();
const PORT = 8000;

app.use(express.json());

app.post("/predict", (req, res) => {
  const input = JSON.stringify(req.body);

  exec(`python3 predict.py '${input}'`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Exec error: ${error.message}`);
      return res.status(500).json({ error: "Internal error" });
    }

    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (err) {
      console.error("Parsing error:", err);
      res.status(500).json({ error: "Failed to parse prediction output" });
    }
  });
});

app.get("/", (req, res) => {
  res.send("ML API is running!");
});

app.listen(PORT, () => {
  console.log(`Server listening on: http://localhost:${PORT}`);
});
