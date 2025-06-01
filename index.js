const express = require("express");
const { exec } = require("child_process");
const app = express();
const PORT = 8000;

app.use(express.json());

const { spawn } = require("child_process");

app.post("/predict", (req, res) => {
  const py = spawn("python3", ["predict.py"]);

  let output = "";

  py.stdout.on("data", (data) => {
    output += data.toString();
  });

  py.stderr.on("data", (data) => {
    console.error("stderr:", data.toString());
  });

  py.on("close", (code) => {
    console.log("Raw Python output:", output);  // 👈 Add this
    try {
      const result = JSON.parse(output);
      res.json(result);
    } catch (err) {
      console.error("Parsing error:", err);
      res.status(500).json({ error: "Failed to parse prediction output" });
    }
  });


  // Send request body to Python via stdin
  py.stdin.write(JSON.stringify(req.body));
  py.stdin.end();
});


app.get("/", (req, res) => {
  res.send("ML API is running!");
});

app.listen(PORT, () => {
  console.log(`Server listening on: http://localhost:${PORT}`);
});
