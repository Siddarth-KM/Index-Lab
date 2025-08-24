// --- Imports ---
import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, Calendar, TrendingUp, Settings, Play, Info, AlertCircle, CheckCircle, Brain, Target, MessageSquare, Newspaper, Building } from 'lucide-react';
import ReactDOM from 'react-dom';

// --- Main App Component ---
const StockPredictionApp = () => {
  // --- State Management ---
  const [formData, setFormData] = useState({
    index: 'SPY',
    numStocks: 10,
    startDate: '2024-01-01',
    customStartDate: '',
    predictionWindow: 5,
    confidenceInterval: 70,
    customConfidence: '',
    modelSelection: 'auto',
    selectedModels: [],
    showType: 'top',
  });
  const [showCustomDate, setShowCustomDate] = useState(false);
  const [showCustomConfidence, setShowCustomConfidence] = useState(false);
  const [showModelSelection, setShowModelSelection] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const [showGraphModal, setShowGraphModal] = useState(false);
  const [openTooltip, setOpenTooltip] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0 });
  const iconRefs = useRef({});
  const [customTicker, setCustomTicker] = useState('');
  const isCustom = formData.index === 'CUSTOM';
  // ...existing code...

  // --- Effect: Close modal on ESC key ---
  useEffect(() => {
    if (!showGraphModal) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setShowGraphModal(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showGraphModal]);

  // --- Effect: Click-away handler for tooltip ---
  useEffect(() => {
    if (!openTooltip) return;
    function handleClick(e) {
      const ref = iconRefs.current[openTooltip];
      if (
        ref &&
        !ref.contains(e.target) &&
        document.getElementById('prediction-tooltip') &&
        !document.getElementById('prediction-tooltip').contains(e.target)
      ) {
        setOpenTooltip(null);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [openTooltip]);

  // --- Options for dropdowns and models ---
  const indexes = [
    { value: 'SPY', label: 'SPY - S&P 500' },
    { value: 'DOW', label: 'Dow Jones' },
    { value: 'NASDAQ', label: 'NASDAQ' },
    { value: 'SP400', label: 'S&P 400 MidCap' },
    { value: 'SPLV', label: 'SPLV - Low Volatility' },
    { value: 'SPHB', label: 'SPHB - High Beta' },
    { value: 'CUSTOM', label: 'Single Ticker' }
  ];
  const startDateOptions = [
    { value: '2023-01-01', label: '2023-01-01' },
    { value: '2024-01-01', label: '2024-01-01' },
    { value: '2025-01-01', label: '2025-01-01' },
    { value: 'custom', label: 'Custom Date' }
  ];
  const confidenceOptions = [
    { value: '60', label: '60%' },
    { value: '70', label: '70%' },
    { value: '80', label: '80%' },
    { value: '90', label: '90%' },
    { value: 'custom', label: 'Custom' }
  ];
  const modelSelectionOptions = [
    { value: 'auto', label: 'Auto Select (Market Based)' },
    { value: 'manual', label: 'Manual Selection' }
  ];
  const showTypeOptions = [
    { value: 'top', label: 'Top N Predictions' },
    { value: 'worst', label: 'Worst N Predictions' }
  ];
  const availableModels = [
    { id: 1, name: 'XGBoost Quantile Regression', description: 'Low vol, precise' },
    { id: 2, name: 'Random Forest Bootstrap', description: 'Balanced' },
    { id: 3, name: 'Neural Network Conformal', description: 'High vol, complex' },
    { id: 4, name: 'Extra Trees Bootstrap', description: 'High vol, aggressive' },
    { id: 5, name: 'AdaBoost Conformal', description: 'High vol, adaptive' },
    { id: 6, name: 'Bayesian Ridge Conformal', description: 'Low vol, conservative' },
    { id: 7, name: 'Support Vector Regression', description: 'Balanced, stable' },
    { id: 8, name: 'Gradient Boosting Conformal', description: 'High vol, adaptive' },
    { id: 9, name: 'Elastic Net Conformal', description: 'Low vol, regularized' },
    { id: 10, name: 'MLPRegressor Sequence Model', description: 'Sequence modeling (scikit-learn MLPRegressor)' }
  ];
  // --- Model explanations for tooltips ---
  const modelExplanations = {
    1: {
      title: 'XGBoost Quantile Regression',
      desc: 'Gradient boosting for quantile prediction. Excels in low volatility, precise forecasts. Good for stable indexes.'
    },
    2: {
      title: 'Random Forest Bootstrap',
      desc: 'Ensemble of decision trees with bootstrapping. Balanced, robust to noise. Works well in most markets.'
    },
    3: {
      title: 'Neural Network Conformal',
      desc: 'Deep learning with conformal prediction. Handles high volatility and complex patterns. Best for tech/growth stocks.'
    },
    4: {
      title: 'Extra Trees Bootstrap',
      desc: 'Randomized trees for aggressive learning. Suited for high volatility and rapid market shifts.'
    },
    5: {
      title: 'AdaBoost Conformal',
      desc: 'Adaptive boosting with conformal intervals. Adapts to changing conditions, good for volatile or trending markets.'
    },
    6: {
      title: 'Bayesian Ridge Conformal',
      desc: 'Bayesian regression with conformal intervals. Conservative, best for low volatility or mean-reverting indexes.'
    },
    7: {
      title: 'Support Vector Regression',
      desc: 'Margin-based regression. Stable, good for balanced or sideways markets.'
    },
    8: {
      title: 'Gradient Boosting Conformal',
      desc: 'Boosted trees with conformal intervals. Adapts to volatility, good for dynamic markets.'
    },
    9: {
      title: 'Elastic Net Conformal',
      desc: 'Regularized regression with conformal intervals. Best for low volatility, regularized for stability.'
    },
    10: {
      title: 'MLPRegressor Sequence Model',
      desc: 'Sequence modeling using scikit-learn MLPRegressor. Captures sequential patterns, but is not a deep learning transformer.'
    }
  };

  // --- Handlers for form and model selection ---
  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    // Reset results and error on any config change
    setResults(null);
    setError(null);
    setLoading(false);
    if (field === 'index') {
      if (value === 'CUSTOM') {
        setCustomTicker('');
      }
    }
    if (field === 'startDate') {
      setShowCustomDate(value === 'custom');
    }
    if (field === 'confidenceInterval') {
      setShowCustomConfidence(value === 'custom');
    }
    if (field === 'modelSelection') {
      setShowModelSelection(value === 'manual');
      if (value !== 'manual') {
        setFormData(prev => ({
          ...prev,
          selectedModels: []
        }));
      }
    }
  };
  const handleModelToggle = (modelId) => {
    setFormData(prev => {
      const currentModels = prev.selectedModels;
      const isSelected = currentModels.includes(modelId);
      if (isSelected) {
        return {
          ...prev,
          selectedModels: currentModels.filter(id => id !== modelId)
        };
      } else if (currentModels.length < 3) {
        return {
          ...prev,
          selectedModels: [...currentModels, modelId]
        };
      }
      return prev;
    });
  };
  // --- Form submit handler ---
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);
    const submitData = {
      index: formData.index,
      ...(formData.index !== 'CUSTOM' && { numStocks: formData.numStocks }),
      ...(formData.index === 'CUSTOM' && { customTicker }),
      startDate: formData.startDate === 'custom' ? formData.customStartDate : formData.startDate,
      predictionWindow: formData.predictionWindow,
      confidenceInterval: formData.confidenceInterval === 'custom' ? parseInt(formData.customConfidence) : parseInt(formData.confidenceInterval),
      modelSelection: formData.modelSelection,
      selectedModels: formData.selectedModels,
      showType: formData.showType,
    };
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(submitData)
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        // Handle detailed error messages from backend
        const errorMessage = data.error || 'Unknown error occurred';
        setError(errorMessage);
        return;
      }

      // Ensure 'close' field is present in index_prediction and stock_predictions, but do not display it yet
      if (data.index_prediction && !('close' in data.index_prediction)) {
        data.index_prediction.close = null;
      }
      if (Array.isArray(data.stock_predictions)) {
        data.stock_predictions = data.stock_predictions.map(sp => ({ ...sp, close: ('close' in sp) ? sp.close : null }));
      }
      setResults(data);
    } catch (err) {
      setError(err.message || '❌ Network error: Could not connect to the prediction server');
    } finally {
      setLoading(false);
    }
  };

  // Only allow manual model selection for custom ticker
  const filteredModelSelectionOptions = modelSelectionOptions;

  // --- Main Render ---
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header Section */}
      <div className="container mx-auto px-4 pt-8 pb-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">
            IndexLab: AI-Driven Financial Prediction Engine
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Advanced machine learning models for predicting stock movements and index performance with confidence intervals
          </p>
        </div>
        </div>
      <div className="container mx-auto px-4 pb-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 min-h-[80vh] items-start">
          {/* --- Left Column: Input Form --- */}
          <div className="lg:col-span-1 flex flex-col justify-start">
            {/* --- Strategy Configuration Form --- */}
            <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center justify-between">
                <span className="flex items-center">
                  <Settings className="mr-3 text-blue-400" />
                  Prediction Configuration
                </span>
                {/* --- Tooltip for Prediction Configuration --- */}
                <span
                  ref={el => (iconRefs.current['config-guide'] = el)}
                  className="ml-2 cursor-pointer"
                  onClick={e => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    setTooltipPos({
                      top: rect.bottom + window.scrollY + 8,
                      left: rect.left + window.scrollX + rect.width / 2
                    });
                    setOpenTooltip(openTooltip === 'config-guide' ? null : 'config-guide');
                  }}
                  tabIndex={0}
                  aria-label="Prediction Configuration Guide"
                >
                  <Info className="text-blue-400" />
                </span>
                {openTooltip === 'config-guide' && ReactDOM.createPortal(
                  <div
                    id="prediction-tooltip"
                    className="z-[9999] w-72 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                    style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-blue-300">Prediction Configuration</span>
                      <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                    </div>
                    <div>
                      Configure your prediction by selecting the index, number of stocks, data start date, prediction window, confidence interval, and model selection. Use Auto Select if you want the code to pick what it thinks the best models are, or Manual to pick specific models.
                    </div>
                  </div>,
                  document.body
                )}
              </h2>
              {/* --- Prediction Form --- */}
              <form onSubmit={handleSubmit}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* --- Index Selection --- */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Index</label>
                    <div className="relative">
                      <select
                        value={formData.index}
                        onChange={(e) => handleInputChange('index', e.target.value)}
                        className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all appearance-none cursor-pointer"
                      >
                        {indexes.map(index => (
                          <option key={index.value} value={index.value} className="bg-slate-900 text-white">
                            {index.label}
                          </option>
                        ))}
                        <option key="SPSM" value="SPSM" className="bg-slate-900 text-white">
                          S&P 600 Small Cap
                        </option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none" size={20} />
                    </div>
                  </div>

                  {/* Number of Stocks or Custom Ticker */}
                  {formData.index !== 'CUSTOM' ? (
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Number of Stocks</label>
                    <input
                      type="number"
                      value={Number.isFinite(formData.numStocks) ? formData.numStocks : ''}
                      onChange={(e) => handleInputChange('numStocks', parseInt(e.target.value))}
                      min="1"
                      max="100"
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    />
                  </div>
                  ) : (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Single Ticker</label>
                      <input
                        type="text"
                        value={customTicker}
                        onChange={e => setCustomTicker(e.target.value.toUpperCase())}
                        placeholder="e.g. AAPL"
                        className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                        maxLength={8}
                        autoCapitalize="characters"
                      />
                    </div>
                  )}

                  {/* Start Date */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center">
                      Data Start Date
                      <span
                        ref={el => (iconRefs.current['data-start'] = el)}
                        className="ml-1 cursor-pointer"
                        onClick={e => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setTooltipPos({
                            top: rect.bottom + window.scrollY + 8,
                            left: rect.left + window.scrollX + rect.width / 2
                          });
                          setOpenTooltip(openTooltip === 'data-start' ? null : 'data-start');
                        }}
                        tabIndex={0}
                        aria-label="Data Start Date Info"
                      >
                        <Info className="text-blue-400 w-4 h-4" />
                      </span>
                      {openTooltip === 'data-start' && ReactDOM.createPortal(
                        <div
                          id="prediction-tooltip"
                          className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                          style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-blue-300">Data Start Date</span>
                            <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                          </div>
                          <div>
                            Earlier start dates use more historical data, which can improve model accuracy but may also include outdated stock patterns that are less relevant to current markets.
                          </div>
                        </div>,
                        document.body
                      )}
                    </label>
                    <div className="relative">
                      <select
                        value={formData.startDate}
                        onChange={(e) => handleInputChange('startDate', e.target.value)}
                        className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all appearance-none cursor-pointer"
                      >
                        {startDateOptions.map(option => (
                          <option key={option.value} value={option.value} className="bg-slate-900 text-white">
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none" size={20} />
                    </div>
                  </div>

                  {/* Custom Start Date */}
                  {showCustomDate && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Custom Start Date</label>
                      <div className="relative">
                        <input
                          type="date"
                          value={formData.customStartDate}
                          onChange={(e) => handleInputChange('customStartDate', e.target.value)}
                          className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                        />
                        <Calendar className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none" size={20} />
                      </div>
                    </div>
                  )}

                  {/* Prediction Window */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Prediction Window (days)</label>
                    <input
                      type="number"
                      value={Number.isFinite(formData.predictionWindow) ? formData.predictionWindow : ''}
                      onChange={(e) => handleInputChange('predictionWindow', parseInt(e.target.value))}
                      min="1"
                      max="365"
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    />
                  </div>

                  {/* Confidence Interval */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center">
                      Confidence Interval
                      <span
                        ref={el => (iconRefs.current['confidence'] = el)}
                        className="ml-1 cursor-pointer"
                        onClick={e => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setTooltipPos({
                            top: rect.bottom + window.scrollY + 8,
                            left: rect.left + window.scrollX + rect.width / 2
                          });
                          setOpenTooltip(openTooltip === 'confidence' ? null : 'confidence');
                        }}
                        tabIndex={0}
                        aria-label="Confidence Interval Info"
                      >
                        <Info className="text-blue-400 w-4 h-4" />
                      </span>
                      {openTooltip === 'confidence' && ReactDOM.createPortal(
                        <div
                          id="prediction-tooltip"
                          className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                          style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-blue-300">Confidence Interval</span>
                            <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                          </div>
                          <div>
                            The confidence interval shows the range in which the true prediction is likely to fall. Higher confidence means a wider range but more certainty; lower confidence is narrower but less certain.
                          </div>
                        </div>,
                        document.body
                      )}
                    </label>
                    <div className="relative">
                      <select
                        value={formData.confidenceInterval}
                        onChange={(e) => handleInputChange('confidenceInterval', e.target.value)}
                        className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all appearance-none cursor-pointer"
                      >
                        {confidenceOptions.map(option => (
                          <option key={option.value} value={option.value} className="bg-slate-900 text-white">
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none" size={20} />
                    </div>
                  </div>

                  {/* Show Type - Only show for non-custom tickers */}
                  {formData.index !== 'CUSTOM' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Show
                      </label>
                      <div className="relative">
                        <select
                          value={formData.showType}
                          onChange={(e) => handleInputChange('showType', e.target.value)}
                          className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all appearance-none cursor-pointer"
                        >
                          {showTypeOptions.map(option => (
                            <option key={option.value} value={option.value} className="bg-slate-900 text-white">
                              {option.label}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none" size={20} />
                      </div>
                    </div>
                  )}

                  {/* Custom Confidence Interval */}
                  {showCustomConfidence && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Custom Confidence Interval (%)</label>
                      <input
                        type="number"
                        value={Number.isFinite(formData.customConfidence) ? formData.customConfidence : ''}
                        onChange={(e) => handleInputChange('customConfidence', e.target.value)}
                        min="50"
                        max="99"
                        className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                        placeholder="Enter custom confidence (50-99)"
                      />
                    </div>
                  )}
                </div>

                {/* Model Selection */}
                <div className="mt-6">
                  <label className="block text-sm font-medium text-gray-300 mb-4 flex items-center">
                    Model Selection
                    <span
                      ref={el => (iconRefs.current['model-select'] = el)}
                      className="ml-1 cursor-pointer"
                      onClick={e => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setTooltipPos({
                          top: rect.bottom + window.scrollY + 8,
                          left: rect.left + window.scrollX + rect.width / 2
                        });
                        setOpenTooltip(openTooltip === 'model-select' ? null : 'model-select');
                      }}
                      tabIndex={0}
                      aria-label="Model Selection Info"
                    >
                      <Info className="text-blue-400 w-4 h-4" />
                    </span>
                    {openTooltip === 'model-select' && ReactDOM.createPortal(
                      <div
                        id="prediction-tooltip"
                        className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                        style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                      >
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-bold text-blue-300">Model Selection</span>
                          <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                        </div>
                        <div>
                          Auto Select chooses the best models for the current market. Manual lets you pick up to 3 models to combine for the prediction.
                        </div>
                      </div>,
                      document.body
                    )}
                  </label>
                  <div className="space-y-3">
                    {filteredModelSelectionOptions.map(option => (
                      <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                        <input
                          type="radio"
                          name="modelSelection"
                          value={option.value}
                          checked={formData.modelSelection === option.value}
                          onChange={(e) => handleInputChange('modelSelection', e.target.value)}
                          className="w-4 h-4 text-blue-400 focus:ring-blue-400 focus:ring-2"
                        />
                        <span className="text-white">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Manual Model Selection */}
                {showModelSelection && (
                  <div className="mt-6">
                    <div className="flex items-center gap-2 mb-4">
                      <label className="text-gray-300 text-sm font-medium">Select up to 3 models:</label>
                      <Info className="text-blue-400" size={16} />
                      <span className="text-blue-400 text-sm">({formData.selectedModels.length}/3)</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {availableModels.map(model => (
                        <label
                          key={model.id}
                          className={`relative flex items-start space-x-3 p-3 rounded-xl border-2 cursor-pointer transition-all ${
                            formData.selectedModels.includes(model.id)
                              ? 'border-blue-400 bg-blue-400/10'
                              : 'border-white/20 bg-white/5 hover:border-white/40'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={formData.selectedModels.includes(model.id)}
                            onChange={() => handleModelToggle(model.id)}
                            disabled={!formData.selectedModels.includes(model.id) && formData.selectedModels.length >= 3}
                            className="w-4 h-4 text-blue-400 focus:ring-blue-400 focus:ring-2 mt-1"
                          />
                          <div className="flex items-center space-x-2 w-full">
                            <div>
                                <div className="text-white font-medium text-sm flex items-center">
                                  {model.name}
                                </div>
                                <div className="text-gray-400 text-xs">{model.description}</div>
                              </div>
                            </div>
                            <span
                              ref={el => (iconRefs.current[`model-${model.id}`] = el)}
                              className="absolute top-2 right-2 cursor-pointer"
                              onClick={e => {
                                const rect = e.currentTarget.getBoundingClientRect();
                                setTooltipPos({
                                  top: rect.bottom + window.scrollY + 8,
                                  left: rect.left + window.scrollX + rect.width / 2
                                });
                                setOpenTooltip(openTooltip === `model-${model.id}` ? null : `model-${model.id}`);
                              }}
                              tabIndex={0}
                              aria-label={`${model.name} Info`}
                            >
                              <Info className="text-blue-400 w-4 h-4" />
                            </span>
                            {openTooltip === `model-${model.id}` && ReactDOM.createPortal(
                              <div
                                id="prediction-tooltip"
                                className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                              >
                                <div className="flex justify-between items-center mb-1">
                                  <span className="font-bold text-blue-300">{modelExplanations[model.id].title}</span>
                                  <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                                </div>
                                <div>{modelExplanations[model.id].desc}</div>
                              </div>,
                              document.body
                            )}
                          </label>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Submit Button */}
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full mt-6 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 disabled:scale-100 flex items-center justify-center"
                  >
                    {loading ? (
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                    ) : (
                      <Play className="mr-3" />
                    )}
                    {loading ? 'Predicting...' : 'Run Prediction'}
                  </button>
                </form>

                {/* Sentiment Analysis Box for All Tickers (Custom and Indexes) */}
                {results && results.sentiment_analysis && (
                  <div className="mt-6 bg-white/12 backdrop-blur-lg rounded-3xl p-6 border border-white/20">
                    <h3 className="text-xl font-bold text-white mb-4 text-center flex items-center justify-center relative">
                      <MessageSquare className="mr-2 text-blue-400" />
                      Sentiment Analysis
                      <span
                        ref={el => (iconRefs.current['sentiment-analysis'] = el)}
                        className="ml-2 cursor-pointer absolute right-0 top-0"
                        onClick={e => {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setTooltipPos({
                            top: rect.bottom + window.scrollY + 8,
                            left: rect.left + window.scrollX + rect.width / 2
                          });
                          setOpenTooltip(openTooltip === 'sentiment-analysis' ? null : 'sentiment-analysis');
                        }}
                        tabIndex={0}
                        aria-label="Sentiment Analysis Info"
                      >
                        <Info className="text-blue-400 w-4 h-4" />
                      </span>
                      {openTooltip === 'sentiment-analysis' && ReactDOM.createPortal(
                        <div
                          id="sentiment-tooltip"
                          className="z-[9999] w-80 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                          style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-blue-300">Sentiment Analysis</span>
                            <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                          </div>
                          <div>
                            {results.sentiment_analysis.sentiment_details?.is_index ? (
                              <>
                                <b>Index Sentiment Analysis</b> with <b>index-specific news</b>:<br/>
                                • <b>Index News (70%)</b>: Articles specifically about the index (e.g., "Dow Jones", "S&P 500", "NASDAQ")<br/>
                                • <b>Market Sentiment (30%)</b>: Overall market conditions from major indices<br/><br/>
                                <b>Search Terms:</b> For indexes like DOW, we search for "Dow Jones", "DJIA", "Industrial Average", etc. to find relevant news.<br/><br/>
                                <b>Impact:</b> Index sentiment combines specific index coverage with broader market conditions to provide comprehensive sentiment analysis.
                              </>
                            ) : (
                              <>
                                <b>Real-time sentiment</b> analysis with <b>dynamic weighting</b>:<br/>
                                • <b>Company News (Base: 50%)</b>: Recent news specific to this stock<br/>
                                • <b>Sector News (Base: 20%)</b>: News affecting the company's industry<br/>
                                • <b>Market News (Base: 30%)</b>: Overall market sentiment from major indices<br/><br/>
                                <b>Dynamic Weighting:</b> When company or sector news is unavailable, their weights are redistributed proportionally to available sources, ensuring market news gets more influence when specific news is missing.<br/><br/>
                                <b>Impact:</b> Strong sentiment can change the direction of technical predictions, helping identify when news sentiment might override technical signals.
                              </>
                            )}
                          </div>
                        </div>,
                        document.body
                      )}
                    </h3>
                    
                    <div className={`grid ${results.sentiment_analysis.sentiment_details?.is_index ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-3'} gap-4`}>
                      {/* Overall Sentiment Score */}
                      <div className="bg-white/5 rounded-xl p-4 border border-white/10 text-center">
                        <div className="flex items-center justify-center mb-2">
                          <Target className="mr-2 text-purple-400" size={20} />
                          <span className="text-sm font-medium text-gray-300">Overall Score</span>
                        </div>
                        <div className={`text-2xl font-bold ${
                          results.sentiment_analysis.sentiment_score > 20 ? 'text-green-400' :
                          results.sentiment_analysis.sentiment_score < -20 ? 'text-red-400' :
                          'text-yellow-400'
                        }`}>
                          {results.sentiment_analysis.sentiment_score > 0 ? '+' : ''}{Math.round(results.sentiment_analysis.sentiment_score)}
                        </div>
                        <div className="text-xs text-gray-400 mt-1">
                          {results.sentiment_analysis.sentiment_score > 20 ? 'Bullish' :
                           results.sentiment_analysis.sentiment_score < -20 ? 'Bearish' :
                           'Neutral'}
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                          <div 
                            className={`h-2 rounded-full ${
                              results.sentiment_analysis.sentiment_score > 0 ? 'bg-green-400' : 'bg-red-400'
                            }`}
                            style={{ 
                              width: `${Math.min(Math.abs(results.sentiment_analysis.sentiment_score), 100)}%`,
                              marginLeft: results.sentiment_analysis.sentiment_score < 0 ? 
                                `${100 - Math.min(Math.abs(results.sentiment_analysis.sentiment_score), 100)}%` : '0'
                            }}
                          ></div>
                        </div>
                      </div>

                      {/* Conditional display based on whether it's an index or single ticker */}
                      {results.sentiment_analysis.sentiment_details?.is_index ? (
                        /* Index: Only show Market Analysis */
                        <div className="bg-white/5 rounded-xl p-4 border border-white/10 text-center">
                          <div className="flex items-center justify-center mb-2">
                            <Newspaper className="mr-2 text-green-400" size={20} />
                            <span className="text-sm font-medium text-gray-300">Market Analysis</span>
                          </div>
                          <div className="text-lg font-semibold text-white">
                            <span className={`${
                              results.sentiment_analysis.sentiment_details?.market_sentiment > 0 ? 'text-green-400' : 
                              results.sentiment_analysis.sentiment_details?.market_sentiment < 0 ? 'text-red-400' : 
                              'text-yellow-400'
                            }`}>
                              {Math.round(results.sentiment_analysis.sentiment_details?.market_sentiment || 0)}
                            </span>
                          </div>
                          <div className="text-xs text-gray-400">Technical Indicators</div>
                          <div className="text-xs text-green-300 mt-1">
                            100% Weight (Market Only)
                          </div>
                        </div>
                      ) : (
                        /* Single Ticker: Show Company and Sector & Market */
                        <>
                          {/* Company News */}
                          <div className="bg-white/5 rounded-xl p-4 border border-white/10 text-center">
                            <div className="flex items-center justify-center mb-2">
                              <Building className="mr-2 text-blue-400" size={20} />
                              <span className="text-sm font-medium text-gray-300">
                                {results.sentiment_analysis.sentiment_details?.is_index ? 'Index' : 'Company'}
                              </span>
                            </div>
                            <div className="text-lg font-semibold text-white">
                              <span className={`${
                                results.sentiment_analysis.sentiment_details?.company_sentiment_score > 0 ? 'text-green-400' : 
                                results.sentiment_analysis.sentiment_details?.company_sentiment_score < 0 ? 'text-red-400' : 
                                'text-yellow-400'
                              }`}>
                                {results.sentiment_analysis.sentiment_details?.company_sentiment_score > 0 ? '+' : ''}
                                {Math.round(results.sentiment_analysis.sentiment_details?.company_sentiment_score || 0)}
                              </span>
                            </div>
                            <div className="text-xs text-gray-400">Sentiment Score</div>
                            <div className="text-xs text-blue-300 mt-1">
                              {results.sentiment_analysis.sentiment_details?.has_company_news ? 
                                `${Math.round((results.sentiment_analysis.sentiment_details?.company_weight || 0) * 100)}% Weight` : 
                                'No News (0%)'
                              }
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              {results.sentiment_analysis.sentiment_details?.company_articles || 0} {results.sentiment_analysis.sentiment_details?.is_index ? 'index' : ''} articles analyzed
                            </div>
                          </div>

                          {/* Sector & Market (or just Market for indexes) */}
                          <div className="bg-white/5 rounded-xl p-4 border border-white/10 text-center">
                            <div className="flex items-center justify-center mb-2">
                              <Newspaper className="mr-2 text-green-400" size={20} />
                              <span className="text-sm font-medium text-gray-300">
                                {results.sentiment_analysis.sentiment_details?.is_index ? 'Market' : 'Sector & Market'}
                              </span>
                            </div>
                            <div className="text-sm text-white">
                              {results.sentiment_analysis.sentiment_details?.is_index ? (
                                /* Index: Show only Market sentiment */
                                <div className="flex justify-between">
                                  <span>Market:</span>
                                  <span className={`${
                                    results.sentiment_analysis.sentiment_details?.market_sentiment > 0 ? 'text-green-400' : 
                                    results.sentiment_analysis.sentiment_details?.market_sentiment < 0 ? 'text-red-400' : 
                                    'text-yellow-400'
                                  }`}>
                                    {results.sentiment_analysis.sentiment_details?.market_sentiment > 0 ? '+' : ''}
                                    {Math.round(results.sentiment_analysis.sentiment_details?.market_sentiment || 0)}
                                  </span>
                                </div>
                              ) : (
                                /* Single Ticker: Show both Sector and Market */
                                <>
                                  <div className="flex justify-between">
                                    <span>Sector:</span>
                                    <span className={`${
                                      results.sentiment_analysis.sentiment_details?.sector_sentiment_score > 0 ? 'text-green-400' : 
                                      results.sentiment_analysis.sentiment_details?.sector_sentiment_score < 0 ? 'text-red-400' : 
                                      'text-yellow-400'
                                    }`}>
                                      {results.sentiment_analysis.sentiment_details?.sector_sentiment_score > 0 ? '+' : ''}
                                      {Math.round(results.sentiment_analysis.sentiment_details?.sector_sentiment_score || 0)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Market:</span>
                                    <span className={`${
                                      results.sentiment_analysis.sentiment_details?.market_sentiment > 0 ? 'text-green-400' : 
                                      results.sentiment_analysis.sentiment_details?.market_sentiment < 0 ? 'text-red-400' : 
                                      'text-yellow-400'
                                    }`}>
                                      {results.sentiment_analysis.sentiment_details?.market_sentiment > 0 ? '+' : ''}
                                      {Math.round(results.sentiment_analysis.sentiment_details?.market_sentiment || 0)}
                                    </span>
                                  </div>
                                </>
                              )}
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              {results.sentiment_analysis.sentiment_details?.is_index ? (
                                /* Index: Show only market weight */
                                `Market: ${Math.round((results.sentiment_analysis.sentiment_details?.market_weight || 0) * 100)}%`
                              ) : (
                                /* Single Ticker: Show sector + market weights */
                                `Sector: ${Math.round((results.sentiment_analysis.sentiment_details?.sector_weight || 0) * 100)}% + Market: ${Math.round((results.sentiment_analysis.sentiment_details?.market_weight || 0) * 100)}%`
                              )}
                            </div>
                            {!results.sentiment_analysis.sentiment_details?.is_index && (
                              <div className="text-xs text-gray-400 mt-1">
                                {results.sentiment_analysis.sentiment_details?.sector_articles || 0} sector articles analyzed
                              </div>
                            )}
                          </div>
                        </>
                      )}
                    </div>

                    {/* Sector Information - Only show for non-indexes */}
                    {results.sentiment_analysis.sentiment_details?.sector && 
                     !results.sentiment_analysis.sentiment_details?.is_index && (
                      <div className="mt-4 text-center">
                        <span className="text-sm text-gray-300">Industry: </span>
                        <span className="text-sm font-medium text-blue-300">
                          {results.sentiment_analysis.sentiment_details.sector}
                        </span>
                      </div>
                    )}

                    {/* Prediction Impact - Now shows for both single tickers and indexes */}
                    {results.sentiment_analysis.original_ml_prediction !== undefined && (
                      <div className="mt-4 p-3 bg-white/5 rounded-lg border border-white/10">
                        <div className="text-sm text-gray-300 text-center mb-2">Sentiment Impact on Prediction</div>
                        <div className="flex justify-center items-center space-x-4 text-sm">
                          <div className="text-center">
                            <div className="text-xs text-gray-400">ML Only</div>
                            <div className="font-medium text-blue-300">
                              {(results.sentiment_analysis.original_ml_prediction * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div className="text-gray-400">→</div>
                          <div className="text-center">
                            <div className="text-xs text-gray-400">With Sentiment</div>
                            <div className={`font-medium ${
                              results.index_prediction.pred > results.sentiment_analysis.original_ml_prediction ? 'text-green-400' :
                              results.index_prediction.pred < results.sentiment_analysis.original_ml_prediction ? 'text-red-400' :
                              'text-yellow-400'
                            }`}>
                              {(results.index_prediction.pred * 100).toFixed(2)}%
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {error && (
                  <div className="mt-4 p-6 bg-red-500/20 border border-red-500/30 rounded-xl">
                    <div className="flex items-start">
                      <AlertCircle className="mr-3 text-red-400 mt-1 flex-shrink-0" />
                      <div className="flex-1">
                        <h4 className="text-red-200 font-semibold mb-2">❌ Prediction Error</h4>
                        <p className="text-red-200 text-sm leading-relaxed">{error}</p>
                        <div className="mt-3 text-red-300 text-xs">
                          <strong>Possible solutions:</strong>
                          <ul className="mt-1 ml-4 list-disc">
                            <li>Check your internet connection</li>
                            <li>Verify the ticker symbol is correct</li>
                            <li>Try a different date range</li>
                            <li>Reduce the number of stocks</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              {/* Models Used and Market Analysis for Indexes (non-CUSTOM) */}
              {results && !isCustom && results.selected_models && results.selected_models.length > 0 && (
                <div className="mt-8 bg-white/10 backdrop-blur-lg rounded-3xl p-6 border border-blue-400/20">
                  <h3 className="text-xl font-bold text-white mb-4 text-center">Models Used for Prediction</h3>
                  <div className="flex flex-wrap justify-center gap-3">
                    {results.selected_models.map((modelId) => {
                      const model = availableModels.find(m => m.id === modelId) || { name: `Model ${modelId}` };
                      return (
                        <span key={modelId} className="px-4 py-2 bg-blue-500/20 text-blue-200 rounded-lg font-semibold text-sm border border-blue-400/30">
                          {model.name}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
              {results && !isCustom && results.market_condition && (
                <div className="mt-8 bg-white/10 backdrop-blur-lg rounded-3xl p-6 border border-white/20">
                  <h3 className="text-xl font-bold text-white mb-4 text-center flex items-center justify-center relative">
                    Market Analysis
                    <span
                      ref={el => (iconRefs.current['market-analysis'] = el)}
                      className="ml-2 cursor-pointer absolute right-0 top-0"
                      onClick={e => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setTooltipPos({
                          top: rect.bottom + window.scrollY + 8,
                          left: rect.left + window.scrollX + rect.width / 2
                        });
                        setOpenTooltip(openTooltip === 'market-analysis' ? null : 'market-analysis');
                      }}
                      tabIndex={0}
                      aria-label="Market Analysis Info"
                    >
                      <Info className="text-blue-400 w-4 h-4" />
                    </span>
                    {openTooltip === 'market-analysis' && ReactDOM.createPortal(
                      <div
                        id="prediction-tooltip"
                        className="z-[9999] w-72 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                        style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                      >
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-bold text-blue-300">Market Types</span>
                          <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                        </div>
                        <div>
                          <b>Bull:</b> Strong upward trend, models favor growth stocks.<br/>
                          <b>Bear:</b> Downward trend, models become more conservative.<br/>
                          <b>Sideways:</b> No clear trend, models may reduce risk or avoid trades.<br/>
                          <b>Volatile:</b> High price swings, models use wider confidence intervals.<br/>
                          <b>Effect:</b> The market type influences which models are chosen and how predictions are weighted.
                        </div>
                      </div>,
                      document.body
                    )}
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">Market Condition</div>
                      <div className="text-lg font-bold text-white capitalize">{results.market_condition}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">Market Strength</div>
                      <div className="text-lg font-bold text-white">{(results.market_strength * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              )}
            </div> {/* End of left column */}

            {/* Results Area (moved out of left column) */}
            <div className="lg:col-span-1 flex flex-col justify-end">
              {/* Results Display */}
              {results && (
                <>
                  {console.log('Prediction results:', results)}
                  
                  {/* System Messages Display */}
                  {results.system_messages && results.system_messages.length > 0 && (
                    <div className="w-full space-y-3 mb-6">
                      {results.system_messages.map((msg, index) => (
                        <div 
                          key={index} 
                          className={`p-4 rounded-xl border ${
                            msg.type === 'warning' 
                              ? 'bg-yellow-500/10 border-yellow-400/30 text-yellow-200' 
                              : 'bg-blue-500/10 border-blue-400/30 text-blue-200'
                          }`}
                        >
                          <div className="flex items-center">
                            <span className="mr-2">
                              {msg.type === 'warning' ? '⚠️' : 'ℹ️'}
                            </span>
                            <span className="text-sm font-medium">{msg.message}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* Show warning if any key result is missing */}
                <div className="w-full space-y-6">
                    {isCustom ? (
                      <>
                        {/* Single Ticker Prediction Box */}
                        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                          <h3 className="text-2xl font-bold text-white mb-6 text-center">Single Ticker Prediction</h3>
                          <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl p-6 border border-blue-400/30">
                            <div className="text-center">
                              <div className="text-3xl font-bold text-white mb-2">{(results.index_prediction && (results.index_prediction.index_name || results.index_prediction.ticker)) || '--'}</div>
                              <div className="text-4xl font-bold text-green-400 mb-2">
                                {results.index_prediction && results.index_prediction.pred !== undefined ? (results.index_prediction.pred * 100).toFixed(2) + '%' : '--'}
                              </div>
                              <div className="text-gray-300 text-sm">
                                Confidence Interval: {results.index_prediction && results.index_prediction.lower !== undefined && results.index_prediction.upper !== undefined ? `${(results.index_prediction.lower * 100).toFixed(2)}% - ${(results.index_prediction.upper * 100).toFixed(2)}%` : '--'}
                              </div>
                              
                              {/* Directional Probability Display for Single Ticker */}
                              {results.index_prediction && results.index_prediction.direction !== undefined && results.index_prediction.direction_probability !== undefined && (
                                <div className="mt-4 pt-4 border-t border-gray-600/50">
                                  <div className="text-sm">
                                    <span className="text-gray-300">Direction Probability: </span>
                                    <span className={`font-semibold ${results.index_prediction.direction === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                                      {results.index_prediction.direction === 'up' ? '↑' : '↓'} {Math.min(100, Math.max(0, results.index_prediction.direction_probability)).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              )}
                              
                              <div className="text-gray-300 text-sm mt-2">
                                Last Close: {results.index_prediction && results.index_prediction.close !== null ? `$${results.index_prediction.close.toFixed(2)}` : '--'}
                              </div>
                            </div>
                          </div>
                        </div>
                        {/* Market Analysis for Single Ticker (only one, between prediction and graph) */}
                        {results.market_condition && (
                          <div className="mt-8 bg-white/10 backdrop-blur-lg rounded-3xl p-6 border border-white/20">
                            <h3 className="text-xl font-bold text-white mb-4 text-center flex items-center justify-center relative">
                              Market Analysis
                              <span
                                ref={el => (iconRefs.current['market-analysis'] = el)}
                                className="ml-2 cursor-pointer absolute right-0 top-0"
                                onClick={e => {
                                  const rect = e.currentTarget.getBoundingClientRect();
                                  setTooltipPos({
                                    top: rect.bottom + window.scrollY + 8,
                                    left: rect.left + window.scrollX + rect.width / 2
                                  });
                                  setOpenTooltip(openTooltip === 'market-analysis' ? null : 'market-analysis');
                                }}
                                tabIndex={0}
                                aria-label="Market Analysis Info"
                              >
                                <Info className="text-blue-400 w-4 h-4" />
                              </span>
                              {openTooltip === 'market-analysis' && ReactDOM.createPortal(
                                <div
                                  id="prediction-tooltip"
                                  className="z-[9999] w-72 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                  style={{ position: 'absolute', top: tooltipPos.top, left: tooltipPos.left, transform: 'translate(-50%, 0)' }}
                                >
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold text-blue-300">Market Types</span>
                                    <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>&times;</button>
                                  </div>
                                  <div>
                                    <b>Bull:</b> Strong upward trend, models favor growth stocks.<br/>
                                    <b>Bear:</b> Downward trend, models become more conservative.<br/>
                                    <b>Sideways:</b> No clear trend, models may reduce risk or avoid trades.<br/>
                                    <b>Volatile:</b> High price swings, models use wider confidence intervals.<br/>
                                    <b>Effect:</b> The market type influences which models are chosen and how predictions are weighted.
                                  </div>
                                </div>,
                                document.body
                              )}
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="text-center">
                                <div className="text-sm text-gray-400 mb-1">Market Condition</div>
                                <div className="text-lg font-bold text-white capitalize">{results.market_condition}</div>
                              </div>
                              <div className="text-center">
                                <div className="text-sm text-gray-400 mb-1">Market Strength</div>
                                <div className="text-lg font-bold text-white">{(results.market_strength * 100).toFixed(1)}%</div>
                              </div>
                            </div>
                          </div>
                        )}
                        {/* Graph for Single Ticker with View Full Size button */}
                        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                          <h3 className="text-2xl font-bold text-white mb-6 text-center">Prediction Analysis Chart</h3>
                          <div className="flex flex-col items-center">
                            {results.plot_image ? (
                              <>
                                <img
                                  src={`data:image/png;base64,${results.plot_image}`}
                                  alt="Prediction Analysis"
                                  className="max-w-full h-auto rounded-lg shadow-lg"
                                  style={{ maxHeight: '500px' }}
                                />
                                <button
                                  className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition"
                                  onClick={() => setShowGraphModal(true)}
                                >
                                  View Full Size
                                </button>
                              </>
                            ) : (
                              <div className="text-gray-400">No chart data available.</div>
                            )}
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        {/* Index Prediction (always render, show placeholder if missing) */}
                    <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                      <h3 className="text-2xl font-bold text-white mb-6 text-center">Index Prediction</h3>
                      <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl p-6 border border-blue-400/30">
                        <div className="text-center">
                              <div className="text-3xl font-bold text-white mb-2">{(results.index_prediction && (results.index_prediction.index_name || results.index_prediction.ticker)) || '--'}</div>
                          <div className="text-4xl font-bold text-green-400 mb-2">
                                {results.index_prediction && results.index_prediction.pred !== undefined ? (results.index_prediction.pred * 100).toFixed(2) + '%' : '--'}
                          </div>
                          <div className="text-gray-300 text-sm">
                                Confidence Interval: {results.index_prediction && results.index_prediction.lower !== undefined && results.index_prediction.upper !== undefined ? `${(results.index_prediction.lower * 100).toFixed(2)}% - ${(results.index_prediction.upper * 100).toFixed(2)}%` : '--'}
                          </div>
                          
                          {/* Directional Probability Display for Index Prediction */}
                          {results.index_prediction && results.index_prediction.direction !== undefined && results.index_prediction.direction_probability !== undefined && (
                            <div className="mt-4 pt-4 border-t border-gray-600/50">
                              <div className="text-sm">
                                <span className="text-gray-300">Direction Probability: </span>
                                <span className={`font-semibold ${results.index_prediction.direction === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                                  {results.index_prediction.direction === 'up' ? '↑' : '↓'} {Math.min(100, Math.max(0, results.index_prediction.direction_probability)).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          )}
                          
                          <div className="text-gray-300 text-sm mt-2">
                                Last Close: {results.index_prediction && results.index_prediction.close !== null ? `$${results.index_prediction.close.toFixed(2)}` : '--'}
                          </div>
                      </div>
                    </div>
                    </div>
                        {/* Stock Predictions (move above graph) */}
                  {results.stock_predictions && results.stock_predictions.length > 0 && (
                    <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                      <h3 className="text-2xl font-bold text-white mb-6 text-center">
                        {formData.showType === 'top' ? 'Top' : 'Worst'} Stock Predictions
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {results.stock_predictions.map((stock, index) => (
                          <div key={stock.ticker} className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 rounded-xl p-4 border border-white/20 hover:border-blue-400/50 transition-all">
                            <div className="text-center">
                              <div className="text-lg font-bold text-white mb-1">{stock.ticker}</div>
                              <div className={`text-2xl font-bold mb-1 ${stock.pred > 0 ? 'text-green-400' : 'text-red-400'}`}>{(stock.pred * 100).toFixed(2)}%</div>
                              <div className="text-xs text-gray-400">{(stock.lower * 100).toFixed(1)}% - {(stock.upper * 100).toFixed(1)}%</div>
                              
                              {/* Directional Probability Display */}
                              {stock.direction !== undefined && stock.direction_probability !== undefined && (
                                <div className="mt-2 pt-2 border-t border-gray-600/50">
                                  <div className="text-xs">
                                    <span className="text-gray-400">Direction Probability: </span>
                                    <span className={`font-semibold ${stock.direction === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                                      {stock.direction === 'up' ? '↑' : '↓'} {Math.min(100, Math.max(0, stock.direction_probability)).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              )}
                              
                              <div className="text-xs text-gray-400 mt-1">
                                Last Close: {stock.close !== null ? `$${stock.close.toFixed(2)}` : '--'}
                              </div>
                              <div className="text-xs text-blue-300">Rank #{index + 1}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                        {/* Graph (now below stock predictions) */}
                    <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                      <h3 className="text-2xl font-bold text-white mb-6 text-center">Prediction Analysis Chart</h3>
                      <div className="flex flex-col items-center">
                            {results.plot_image ? (
                              <>
                        <img 
                          src={`data:image/png;base64,${results.plot_image}`} 
                          alt="Prediction Analysis" 
                          className="max-w-full h-auto rounded-lg shadow-lg"
                          style={{ maxHeight: '500px' }}
                        />
                        <button
                                  className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition"
                          onClick={() => setShowGraphModal(true)}
                        >
                          View Full Size
                        </button>
                      </>
                    ) : (
                      <div className="text-gray-400">No chart data available.</div>
                    )}
                      </div>
                    </div>
                      </>
                  )}
                </div>
                </>
              )}

              {/* Error or Ready Message */}
              {!results && (
                <div className="flex flex-col justify-center items-center h-full">
                  <TrendingUp className="mx-auto mb-6 text-gray-400" size={64} />
                  <h3 className="text-2xl font-bold text-white mb-4">Ready to Predict</h3>
                  <p className="text-gray-300 max-w-md mx-auto">
                    Configure your prediction parameters and run the analysis to see detailed stock movement forecasts.
                  </p>
              </div>
              )}
            </div> {/* End of right column */}
          </div>
        </div>
        {/* Render fullscreen modal at the root using a portal */}
        {showGraphModal && ReactDOM.createPortal(
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90 w-screen h-screen">
            <button
              className="absolute top-6 right-8 text-white text-4xl font-bold hover:text-blue-400 z-60"
              onClick={() => setShowGraphModal(false)}
              aria-label="Close Fullscreen Graph"
            >
              &times;
            </button>
            <img
              src={`data:image/png;base64,${results.plot_image}`}
              alt="Full Size Prediction Analysis"
              className="w-full h-full object-contain"
              style={{ maxWidth: '100vw', maxHeight: '100vh' }}
            />
          </div>,
          document.body
        )}
      </div>
    );
};

  export default StockPredictionApp;