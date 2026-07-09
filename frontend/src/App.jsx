import { useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const starterReviews = [
  'This coffee maker is absolutely amazing and the best thing ever. You need to buy it right now before it sells out! Must buy!',
  'I bought this desk lamp last week and it arrived well packaged. The brightness is steady and the build quality feels solid.',
];

function App() {
  const [review, setReview] = useState(starterReviews[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
            A full-stack detector driven by TF-IDF text features, linguistic heuristics, and an ensemble of ML models.
          </p>
        </div>
      </header>

      <main className="dashboard">
        <section className="panel">
          <h2>Review analyzer</h2>
          <textarea
            rows="8"
            value={review}
            onChange={(event) => setReview(event.target.value)}
            placeholder="Paste a review to inspect..."
          />
          <div className="actions">
            <button onClick={analyze} disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze review'}
            </button>
            <div className="sample-links">
              {starterReviews.map((item, index) => (
                <button key={item} className="secondary" onClick={() => setReview(item)}>
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
              <p className="score">
                Fake probability: <strong>{(result.probability_fake * 100).toFixed(1)}%</strong>
              </p>
              <p className="score">
                Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
              </p>
              <p className="hint">Model accuracy on held-out test data: {result.model_accuracy}</p>
              <div className="feature-grid">
                {Object.entries(result.features || {}).map(([key, value]) => (
                  <div key={key} className="feature-card">
                    <span>{key}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="placeholder">Submit a review to see a prediction and linguistic indicators.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
