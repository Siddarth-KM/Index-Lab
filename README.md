# IndexLab: AI-Driven Financial Prediction Platform

IndexLab is a full-stack web application that leverages machine learning models and a modern React frontend to deliver actionable financial predictions for stocks and indices. The backend, built with Flask and Python, integrates ensemble ML models, technical indicators, and sentiment analysis to provide robust forecasts with confidence intervals. The frontend offers an interactive, responsive UI powered by React and Tailwind CSS, enabling users to visualize predictions, explore market conditions, and analyze custom tickers.

## Key Features
- Multi-model ML backend (XGBoost, Random Forest, Neural Network, etc.)
- Sentiment analysis and technical indicators
- Cross-asset and custom ticker support
- Confidence intervals and market condition detection
- Interactive charts and real-time data
- Caching, threading, and parallel processing for performance
- Modern React frontend with Tailwind CSS

## Project Summary
IndexLab combines advanced machine learning with a user-friendly web interface to empower investors and analysts. The backend orchestrates multiple predictive models, adapts to market regimes, and delivers insights with statistical rigor. The frontend visualizes these insights, making complex analytics accessible to all users. Whether for professional research or personal investing, IndexLab streamlines financial prediction and analysis in a single, integrated platform.

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
| MLPRegressor Sequence Model | MLPRegressor | Sequence modeling |
