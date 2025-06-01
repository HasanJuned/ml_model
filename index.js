const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
app.use(bodyParser.json());

app.post('/predict', (req, res) => {
  const inputData = req.body;

  // Spawn Python process
  const py = spawn('python3', ['predict.py']);

  let dataString = '';

  py.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  py.stderr.on('data', (data) => {
    console.error('Python stderr:', data.toString());
  });


  py.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: 'Python script failed' });
    }
    try {
      const result = JSON.parse(dataString);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: 'Failed to parse Python output' });
    }
  });

  // Send input JSON to Python stdin
  py.stdin.write(JSON.stringify(inputData));
  py.stdin.end();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Node.js server running on port ${PORT}`);
});
