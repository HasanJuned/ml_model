const express = require('express');
const { spawn } = require('child_process');

const app = express();
app.use(express.json());

app.post('/predict', (req, res) => {
  const { age, bmi, trestbps, chol, sex, cp, diabetes, smoker } = req.body;

  // Validate inputs (basic)
  if ([age, bmi, trestbps, chol, sex, cp, diabetes, smoker].some(v => v === undefined)) {
    return res.status(400).json({ error: 'Missing one or more input parameters' });
  }

  // Spawn python process with inputs as args
  const py = spawn('python3', [
    'predict.py',
    age, bmi, trestbps, chol,
    sex, cp, diabetes, smoker
  ].map(String));

  let output = '';
  let errorOutput = '';

  py.stdout.on('data', (data) => {
    output += data.toString();
  });

  py.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  py.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: 'Python script failed', details: errorOutput });
    }
    // output should be prediction 0 or 1
    res.json({ prediction: output.trim() });
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
