# IndexLab: AI-Driven Financial Prediction Engine

![IndexLab](https://img.shields.io/badge/IndexLab-AI%20Prediction%20Engine-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![React](https://img.shields.io/badge/React-18+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3+-red)

## 🚀 Overview

IndexLab is a sophisticated financial prediction platform that leverages advanced machine learning models to predict stock movements and index performance with confidence intervals. Built with React frontend and Flask backend, it provides institutional-grade quantitative analysis tools in an easy-to-use interface.

## ✨ Features

- **🤖 Multi-Model AI Prediction**: 10 different machine learning models optimized for various market conditions
- **📊 Market Analysis**: Automatic detection of market conditions (Bull, Bear, Sideways, Volatile)
- **📈 Index & Custom Ticker Support**: Analyze major indices (SPY, DOW, NASDAQ) or individual stocks
- **🎯 Confidence Intervals**: Adjustable prediction confidence levels (60%-90%)
- **📱 Interactive Charts**: Beautiful visualizations of predictions and historical data
- **⚡ Real-time Data**: Live stock data from Yahoo Finance API
- **🎨 Modern UI**: Sleek, responsive design with dark theme

## 🏗️ Architecture

```
IndexLab/
├── frontend/                 # React application
│   ├── src/
│   │   ├── App.js           # Main application component
│   │   ├── index.js         # React entry point
│   │   └── ...
│   ├── public/
│   ├── package.json
│   └── ...
├── backend_server.py         # Flask backend server
├── requirements.txt          # Python dependencies
├── README.md
└── Deployment.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start backend server
python backend_server.py
```
Backend runs on `http://localhost:5000`

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```
Frontend runs on `http://localhost:3000`

## 🤖 Machine Learning Models

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
| Transformer | Deep Learning | Sequence modeling, complex patterns |

## 📊 Market Conditions

- **🐂 Bull Market**: Strong upward trend, models favor growth stocks
- **🐻 Bear Market**: Downward trend, models become more conservative  
- **➡️ Sideways Market**: No clear trend, models may reduce risk
- **📈 Volatile Market**: High price swings, models use wider confidence intervals

## 🔌 API Endpoints

### POST /api/predict
Main prediction endpoint

**Request:**
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

**Response:**
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
Health check endpoint

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **YFinance**: Stock data API
- **Matplotlib**: Chart generation
- **Scikit-learn**: Machine learning

### Frontend
- **React**: UI framework
- **Tailwind CSS**: Styling
- **Lucide React**: Icons
- **Axios**: HTTP client

## 📦 Installation

### Option 1: Clone Repository
```bash
git clone https://github.com/yourusername/IndexLab.git
cd IndexLab
```

### Option 2: Download ZIP
1. Download the ZIP file
2. Extract to your desired location
3. Follow setup instructions above

## 🚀 Deployment

See [Deployment.md](Deployment.md) for detailed deployment instructions including:
- Heroku deployment
- Docker containerization
- VPS/Cloud setup
- Production optimization

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data
- React and Flask communities
- Open source contributors

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Deployment.md](Deployment.md) troubleshooting section
2. Open an issue on GitHub
3. Contact the development team

---

**Disclaimer**: This tool is for educational and research purposes only. Financial predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment decisions.
