import { useState, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const starterReviews = [
  'This coffee maker is absolutely amazing and the best thing ever. You need to buy it right now before it sells out! Must buy! Five stars!',
  'I bought this desk lamp last week and it arrived well packaged. The brightness is steady and the build quality feels solid for the price.',
  'Worst product ever. Broke after one day. Total waste of money. Do not buy this garbage!',
];

const FEATURE_LABELS = {
  word_count: 'Word Count',
  unique_ratio: 'Lexical Diversity',
  punctuation_ratio: 'Punctuation Ratio',
  uppercase_ratio: 'Uppercase Ratio',
  exclamation_count: 'Exclamation Marks',
  question_count: 'Question Marks',
  sentiment_balance: 'Sentiment Balance',
  avg_word_length: 'Avg Word Length',
  sentence_count: 'Sentence Count',
  avg_sentence_length: 'Avg Sentence Length',
  capital_word_ratio: 'Capital Words',
  repeated_word_ratio: 'Word Repetition',
  superlative_count: 'Superlatives',
  review_length: 'Review Length',
  digit_ratio: 'Digit Ratio',
};

function App() {
  const [review, setReview] = useState(starterReviews[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, []);

  const analyze = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Backend unavailable. Start the FastAPI service first.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">FakeGuard</p>
          <h1>Detect fake product reviews in real time</h1>
          <p className="subtitle">
            Trained on 40,000+ real reviews using TF-IDF vectorization, 15+ linguistic features, and a Random Forest + XGBoost + Logistic Regression ensemble.
          </p>
        </div>
      </header>

      {stats && (
        <section className="stats-banner">
          <div className="stat-card">
            <span className="stat-value">{(stats.dataset_size || 40432).toLocaleString()}+</span>
            <span className="stat-label">Reviews Trained</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.f1_score || '\u2014'}</span>
            <span className="stat-label">F1 Score</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.accuracy || '\u2014'}</span>
            <span className="stat-label">Accuracy</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.precision || '\u2014'}</span>
            <span className="stat-label">Precision</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.recall || '\u2014'}</span>
            <span className="stat-label">Recall</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{stats.feature_count || 15}</span>
            <span className="stat-label">Features</span>
          </div>
        </section>
      )}

      <main className="dashboard">
        <section className="panel">
          <h2>Review Analyzer</h2>
          <textarea
            rows="8"
            value={review}
            onChange={(e) => setReview(e.target.value)}
            placeholder="Paste a review to inspect..."
          />
          <div className="actions">
            <button onClick={analyze} disabled={loading}>
              {loading ? 'Analyzing\u2026' : 'Analyze Review'}
            </button>
            <div className="sample-links">
              {starterReviews.map((item, index) => (
                <button key={index} className="secondary" onClick={() => setReview(item)}>
                  Example {index + 1}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="panel result-panel">
          <h2>Prediction</h2>
          {result?.error ? (
            <p className="error">{result.error}</p>
          ) : result ? (
            <>
              <div className={`badge ${result.predicted_label === 'fake' ? 'fake' : 'genuine'}`}>
                {result.predicted_label.toUpperCase()}
              </div>
              <div className="scores-row">
                <p className="score">
                  Fake probability: <strong>{(result.probability_fake * 100).toFixed(1)}%</strong>
                </p>
                <p className="score">
                  Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                </p>
              </div>
              <p className="hint">
                Model accuracy: {result.model_accuracy} &middot; F1: {result.f1_score}
              </p>
              <h3 className="features-heading">Linguistic Features ({Object.keys(result.features || {}).length})</h3>
              <div className="feature-grid">
                {Object.entries(result.features || {}).map(([key, value]) => (
                  <div key={key} className="feature-card">
                    <span className="feature-name">{FEATURE_LABELS[key] || key}</span>
                    <strong className="feature-value">{value}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="placeholder">Submit a review to see a prediction and linguistic indicators.</p>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <p>FakeGuard &middot; TF-IDF + Ensemble ML &middot; Built with FastAPI &amp; React</p>
      </footer>
    </div>
  );
}

export default App;
