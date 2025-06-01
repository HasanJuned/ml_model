const express = require("express");
const { spawn } = require("child_process");
const app = express();
const PORT = 8000;

app.use(express.json());

app.post("/predict", (req, res) => {
  // Run the notebook using jupyter nbconvert
  const py = spawn("jupyter", [
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "--stdout",
    "ml_model.ipynb"
  ]);

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

    // Extract prediction from output (search for JSON)
    try {
      const match = output.match(/"prediction":\s*\d+/);
      if (match) {
        const prediction = parseInt(match[0].split(":")[1]);
        res.json({ prediction });
      } else {
        res.status(500).json({ error: "Prediction not found in notebook output" });
      }
    } catch (err) {
      console.error("Failed to parse notebook output:", output);
      res.status(500).json({ error: "Failed to parse notebook output" });
    }
  });
});

app.get("/", (req, res) => {
  res.send("ML Notebook API is running!");
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
