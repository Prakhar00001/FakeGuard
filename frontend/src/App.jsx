import React, { useState } from 'react';
import './App.css';

// Relative path routes directly to Vercel Serverless Function (/api/predict)
const API_ENDPOINT = '/api/predict';

const EXAMPLES = [
  {
    label: 'Example 1',
    text: 'BEST PRODUCT EVER I HAVE PURCHASED IN MY WHOLE LIFE! AMAZING QUALITY MUST BUY NOW 100% PERFECT REGRET NOTHING BEST DEAL EVER!'
  },
  {
    label: 'Example 2',
    text: 'I ordered this package last Wednesday and it arrived in 3 days. The material feels durable, though the outer packaging was slightly dented on arrival. For the price point, it gets the job done.'
  },
  {
    label: 'Example 3',
    text: 'TOTAL SCAM DO NOT BUY THIS PRODUCT!!! IT BROKE IN 2 MINUTES ABSOLUTE GARBAGE SELLER IS A CHEAT DO NOT TRUST THIS STORE ZERO STARS!'
  }
];

function App() {
  const [reviewText, setReviewText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleAnalyze = async (textToAnalyze) => {
    const text = textToAnalyze || reviewText;
    if (!text.trim()) {
      setError('Please enter a review to analyze.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review_text: text }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('API Error:', err);
      setError('Backend unavailable. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleText) => {
    setReviewText(exampleText);
    handleAnalyze(exampleText);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="logo">FAKEGUARD</h1>
        <h2 className="title">Detect fake product reviews in real time</h2>
        <p className="subtitle">
          Trained on 40,000+ real reviews using TF-IDF vectorization, 15+ linguistic features, and a Random Forest + XGBoost + Logistic Regression ensemble.
        </p>
      </header>

      <main className="main-content">
        <section className="card input-card">
          <h3>Review Analyzer</h3>
          <textarea
            className="review-input"
            rows="8"
            placeholder="Paste a product review here..."
            value={reviewText}
            onChange={(e) => setReviewText(e.target.value)}
          />

          <div className="button-group">
            <button
              className="btn btn-primary"
              onClick={() => handleAnalyze()}
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Review'}
            </button>

            {EXAMPLES.map((ex, index) => (
              <button
                key={index}
                className="btn btn-secondary"
                onClick={() => handleExampleClick(ex.text)}
                disabled={loading}
              >
                {ex.label}
              </button>
            ))}
          </div>
        </section>

        <section className="card output-card">
          <h3>Prediction</h3>

          {error && <p className="error-text">{error}</p>}

          {!result && !error && !loading && (
            <p className="placeholder-text">Enter a review or select an example to see predictions.</p>
          )}

          {loading && <p className="loading-text">Extracting linguistic features & running model...</p>}

          {result && (
            <div className="result-container">
              <div className={`badge ${result.is_fake ? 'badge-fake' : 'badge-genuine'}`}>
                {result.is_fake ? 'FAKE REVIEW DETECTED' : 'GENUINE REVIEW'}
              </div>

              <div className="metric">
                <span className="metric-label">Fake Probability:</span>
                <span className="metric-value">{(result.fake_probability * 100).toFixed(1)}%</span>
              </div>

              <div className="metric">
                <span className="metric-label">Confidence:</span>
                <span className="metric-value">{result.confidence}</span>
              </div>

              {result.linguistic_features && (
                <div className="features-breakdown">
                  <h4>Extracted Linguistic Features</h4>
                  <ul>
                    <li>Word Count: <strong>{result.linguistic_features.word_count}</strong></li>
                    <li>Uppercase Ratio: <strong>{(result.linguistic_features.uppercase_ratio * 100).toFixed(1)}%</strong></li>
                    <li>Exclamation Ratio: <strong>{(result.linguistic_features.exclamation_ratio * 100).toFixed(1)}%</strong></li>
                    <li>Avg Word Length: <strong>{result.linguistic_features.avg_word_length}</strong></li>
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;