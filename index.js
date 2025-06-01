const express = require('express');
const { spawn } = require('child_process');

const app = express();
app.use(express.json());

app.post('/predict', (req, res) => {
  try {
    const input = req.body;
    console.log("Incoming request body:", input);

    const { spawn } = require('child_process');
    const py = spawn('python3', ['ml_model.py', JSON.stringify(input)]);

    let result = '';
    let errorOutput = '';

    py.stdout.on('data', (data) => {
      result += data.toString();
    });

    py.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error(`Python stderr: ${errorOutput}`);
    });

    py.on('close', (code) => {
      console.log(`Python script exited with code ${code}`);
      if (code !== 0) {
        return res.status(500).json({ error: 'Python script error', stderr: errorOutput });
      }
      res.json({ result: result.trim() });
    });

    py.on('error', (err) => {
      console.error("Failed to start Python process:", err);
      res.status(500).json({ error: 'Failed to run model', details: err.message });
    });
  } catch (err) {
    console.error("Caught exception:", err);
    res.status(500).json({ error: 'Internal server error', details: err.message });
  }
});


app.listen(8000, () => {
  console.log('Server running on http://localhost:8000');
});
