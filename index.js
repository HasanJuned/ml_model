const express = require("express");
const { spawn } = require("child_process");
const app = express();
const PORT = 8000;

app.use(express.json());

app.post("/predict", (req, res) => {
  // Run predict.py which already has hardcoded input
  const py = spawn("python3", ["predict.py"]);

  let output = "";
  let errorOutput = "";

  py.stdout.on("data", (data) => {
    output += data.toString();
  });

  py.stderr.on("data", (data) => {
    errorOutput += data.toString();
  });

  py.on("close", (code) => {
    if (errorOutput) {
      console.error("Python stderr:", errorOutput);
    }

    try {
      const result = JSON.parse(output);
      res.json(result); // send back prediction
    } catch (err) {
      console.error("Failed to parse output:", output);
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
