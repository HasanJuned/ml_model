const express = require('express');
const { spawn } = require('child_process');

const app = express();
app.use(express.json());

app.post('/predict', (req, res) => {
  const input = req.body;

  const pythonProcess = spawn('python3', ['ml_model.py', JSON.stringify(input)]);

  let result = '';
  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: 'Model error' });
    }
    res.json({ result: result.trim() });
  });
});

app.listen(3000, () => {
  console.log('Server running on http://localhost:8000');
});
