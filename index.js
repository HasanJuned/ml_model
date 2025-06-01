const axios = require('axios');

async function getPrediction(inputData) {
  const ngrokUrl = 'http://127.0.0.1:5000.ngrok.io/predict'; // replace with actual ngrok url

  try {
    const response = await axios.post(ngrokUrl, inputData);
    console.log('Prediction:', response.data);
  } catch (error) {
    console.error('Error:', error);
  }
}

getPrediction({
  age: 50,
  bmi: 27,
  trestbps: 120,
  chol: 230,
  sex: 1,
  cp: 1,
  diabetes: 0,
  smoker: 0
});
