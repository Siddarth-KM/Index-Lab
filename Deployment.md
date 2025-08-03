# IndexLab: AI-Driven Financial Prediction Engine

## Overview
IndexLab is a full-stack financial prediction platform combining a Python/Flask backend with a React/Tailwind frontend. It uses advanced machine learning models, technical indicators, and sentiment analysis to predict stock movements and index performance, providing confidence intervals and interactive visualizations.

## Features
- Multi-model ML backend (XGBoost, Random Forest, Neural Network, etc.)
- Sentiment analysis and technical indicators
- Cross-asset and custom ticker support
- Confidence intervals and market condition detection
- Interactive charts and real-time data
- Caching, threading, and parallel processing for performance
- Modern React frontend with Tailwind CSS

## Project Structure
```
IndexLab/
├── frontend/                 # React + Tailwind CSS web app
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
├── backend_server.py         # Flask backend server (ML engine)
├── requirements.txt          # Python dependencies
├── README.md
└── Deployment.md
```

## Setup Instructions

### Backend Setup (Python/Flask)
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start backend server:
   ```bash
   python backend_server.py
   ```
   The backend runs at `http://localhost:5000`
3. Verify backend health:
   ```bash
   curl http://localhost:5000/api/health
   ```

### Frontend Setup (React/Tailwind)
1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start development server:
   ```bash
   npm start
   ```
   The frontend runs at `http://localhost:3000`

## API Endpoints (Backend)

### POST /api/predict
Main prediction endpoint that accepts:
```json
{
  "index": "SPY",
  "numStocks": 10,
  "customTicker": "",
  "startDate": "2024-01-01",
  "predictionWindow": 5,
  "confidenceInterval": 70,
  "modelSelection": "auto",
  "selectedModels": []
}
```

Returns:
```json
{
  "index_prediction": {
    "ticker": "SPY",
    "index_name": "SPY",
    "pred": 0.025,
    "lower": 0.015,
    "upper": 0.035
  },
  "selected_models": [2, 7, 6],
  "market_condition": "bull",
  "market_strength": 0.75,
  "plot_image": "base64_encoded_chart",
  "stock_predictions": [...]
}
```

### GET /api/health
Health check endpoint.

## Machine Learning Models
| Model | Type | Best For |
|-------|------|----------|
| XGBoost Quantile Regression | Gradient Boosting | Low volatility, precise predictions |
| Random Forest Bootstrap | Ensemble | Balanced performance |
| Neural Network Conformal | Deep Learning | High volatility, complex patterns |
| Extra Trees Bootstrap | Ensemble | High volatility, aggressive |
| AdaBoost Conformal | Boosting | High volatility, adaptive |
| Bayesian Ridge Conformal | Bayesian | Low volatility, conservative |
| Support Vector Regression | SVM | Balanced, stable |
| Gradient Boosting Conformal | Boosting | High volatility, adaptive |
| Elastic Net Conformal | Regularized | Low volatility, regularized |
| MLPRegressor Sequence Model | MLPRegressor | Sequence modeling using scikit-learn MLPRegressor |

## Market Conditions

- **Bull Market**: Strong upward trend, models favor growth stocks
- **Bear Market**: Downward trend, models become more conservative
- **Sideways Market**: No clear trend, models may reduce risk
- **Volatile Market**: High price swings, models use wider confidence intervals

## Deployment Options

### Local Development
- Backend: `python backend_server.py`
- Frontend: `npm start` (in frontend directory)

### Production Deployment

#### Option 1: Heroku
1. Create `Procfile` in root:
   ```
   web: python backend_server.py
   ```
2. Deploy to Heroku with Python buildpack

#### Option 2: Docker
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["python", "backend_server.py"]
   ```

#### Option 3: VPS/Cloud
1. Install Python 3.9+ and Node.js
2. Set up nginx as reverse proxy
3. Use systemd for process management
4. Configure SSL certificates

## Environment Variables
- `FLASK_ENV`: Set to 'production' for production deployment
- `PORT`: Port for backend server (default: 5000)

## Troubleshooting

### Common Issues
1. **CORS Errors**: Ensure backend CORS is properly configured
2. **Data Fetch Errors**: Check internet connection and Yahoo Finance API status
3. **Chart Generation**: Verify matplotlib backend is set to 'Agg'
4. **Port Conflicts**: Change ports in backend_server.py if needed

### Performance Optimization
- Enable caching for stock data
- Implement rate limiting for API calls
- Use CDN for frontend assets
- Optimize chart generation

## Security Considerations
- Implement API rate limiting
- Add authentication for production use
- Validate all input parameters
- Use HTTPS in production
- Sanitize user inputs

## Contributing
1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License
This project is licensed under the MIT License. 