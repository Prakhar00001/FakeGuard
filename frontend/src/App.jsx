import React, { useState } from 'react';

// Relative API endpoint works seamlessly locally and on live Vercel
const API_ENDPOINT = '/api/predict';

const EXAMPLES = [
  "AMAZING PRODUCT MUST BUY NOW PERFECT QUALITY 100% BEST EVER!",
  "The product arrived on time and works as expected. Build quality is decent for the price.",
  "DO NOT BUY THIS! absolute garbage scam seller zero stars completely broken on arrival!!!"
];

export default function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeReviewText = async (textToAnalyze) => {
    if (!textToAnalyze.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_text: textToAnalyze }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
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

  const handleFormSubmit = (e) => {
    e.preventDefault();
    analyzeReviewText(review);
  };

  const handleExampleClick = (exampleText) => {
    setReview(exampleText);
    analyzeReviewText(exampleText);
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Detect fake product reviews in real time</h1>
        <p style={styles.subtitle}>
          Trained on 40,000+ real reviews using <strong>TF-IDF vectorization</strong>, <strong>15+ linguistic features</strong>, and a <strong>Random Forest + XGBoost + Logistic Regression ensemble</strong>.
        </p>
      </header>

      <main style={styles.mainGrid}>
        {/* Left Column: Input Form */}
        <div style={styles.card}>
          <h2 style={styles.cardHeader}>Review Analyzer</h2>
          <form onSubmit={handleFormSubmit} style={styles.form}>
            <textarea
              rows="7"
              style={styles.textarea}
              placeholder="Paste review text here..."
              value={review}
              onChange={(e) => setReview(e.target.value)}
            />
            <div style={styles.buttonRow}>
              <button
                type="submit"
                disabled={loading || !review.trim()}
                style={loading ? { ...styles.button, ...styles.buttonDisabled } : styles.button}
              >
                {loading ? 'Analyzing...' : 'Analyze Review'}
              </button>
              {EXAMPLES.map((ex, idx) => (
                <button
                  key={idx}
                  type="button"
                  style={styles.exampleBtn}
                  onClick={() => handleExampleClick(ex)}
                >
                  Example {idx + 1}
                </button>
              ))}
            </div>
          </form>
        </div>

        {/* Right Column: Prediction Results */}
        <div style={styles.card}>
          <h2 style={styles.cardHeader}>Prediction</h2>
          
          {error && <div style={styles.errorText}>{error}</div>}

          {!result && !error && (
            <p style={styles.placeholderText}>
              Enter a review on the left or select an example to view real-time classification metrics.
            </p>
          )}

          {result && (
            <div>
              <div
                style={{
                  ...styles.badge,
                  backgroundColor: result.is_fake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                  color: result.is_fake ? '#f87171' : '#4ade80',
                  border: `1px solid ${result.is_fake ? '#ef4444' : '#22c55e'}`,
                }}
              >
                {result.is_fake ? '🚨 Fake Review Detected' : '✅ Genuine Review'}
              </div>

              <div style={styles.metricGrid}>
                <div style={styles.metricItem}>
                  <span style={styles.metricLabel}>Classification Confidence</span>
                  <span style={styles.metricValue}>{result.confidence}</span>
                </div>
                <div style={styles.metricItem}>
                  <span style={styles.metricLabel}>Fake Probability Score</span>
                  <span style={styles.metricValue}>{result.fake_probability}</span>
                </div>
              </div>

              <h4 style={styles.featureHeader}>15+ Linguistic Features Breakdown</h4>
              <div style={styles.featureGrid}>
                {Object.entries(result.linguistic_features || {}).map(([key, value]) => (
                  <div key={key} style={styles.featurePill}>
                    <span>{key.replace('_', ' ')}:</span>{' '}
                    <strong>{typeof value === 'number' ? value.toFixed(3) : value}</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '3rem 1.5rem',
    color: '#f3f4f6',
    backgroundColor: '#0f172a',
    minHeight: '100vh',
    boxSizing: 'border-box',
  },
  header: {
    marginBottom: '2.5rem',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: '800',
    color: '#ffffff',
    margin: '0 0 0.75rem 0',
  },
  subtitle: {
    fontSize: '1.05rem',
    color: '#94a3b8',
    margin: 0,
    maxWidth: '800px',
    lineHeight: '1.6',
  },
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))',
    gap: '2rem',
  },
  card: {
    backgroundColor: '#1e293b',
    padding: '1.75rem',
    borderRadius: '16px',
    border: '1px solid #334155',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)',
  },
  cardHeader: {
    fontSize: '1.35rem',
    fontWeight: '700',
    color: '#f8fafc',
    margin: '0 0 1.25rem 0',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  textarea: {
    width: '100%',
    padding: '1rem',
    borderRadius: '12px',
    border: '1px solid #475569',
    backgroundColor: '#0f172a',
    color: '#f8fafc',
    fontSize: '0.95rem',
    boxSizing: 'border-box',
    outline: 'none',
    resize: 'vertical',
  },
  buttonRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.5rem',
  },
  button: {
    padding: '0.75rem 1.25rem',
    backgroundColor: '#38bdf8',
    color: '#0f172a',
    border: 'none',
    borderRadius: '8px',
    fontSize: '0.95rem',
    fontWeight: '700',
    cursor: 'pointer',
  },
  buttonDisabled: {
    backgroundColor: '#64748b',
    cursor: 'not-allowed',
  },
  exampleBtn: {
    padding: '0.75rem 1rem',
    backgroundColor: '#334155',
    color: '#cbd5e1',
    border: '1px solid #475569',
    borderRadius: '8px',
    fontSize: '0.85rem',
    fontWeight: '600',
    cursor: 'pointer',
  },
  errorText: {
    color: '#f87171',
    fontSize: '0.95rem',
    fontStyle: 'italic',
  },
  placeholderText: {
    color: '#64748b',
    fontSize: '0.95rem',
  },
  badge: {
    display: 'inline-block',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    fontWeight: '700',
    fontSize: '1rem',
    marginBottom: '1.25rem',
  },
  metricGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '1rem',
    marginBottom: '1.5rem',
  },
  metricItem: {
    display: 'flex',
    flexDirection: 'column',
    padding: '1rem',
    backgroundColor: '#0f172a',
    borderRadius: '8px',
    border: '1px solid #334155',
  },
  metricLabel: {
    fontSize: '0.75rem',
    color: '#94a3b8',
    textTransform: 'uppercase',
  },
  metricValue: {
    fontSize: '1.35rem',
    fontWeight: '700',
    color: '#f8fafc',
    marginTop: '0.25rem',
  },
  featureHeader: {
    fontSize: '0.95rem',
    color: '#cbd5e1',
    marginBottom: '0.75rem',
  },
  featureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
    gap: '0.5rem',
  },
  featurePill: {
    padding: '0.4rem 0.6rem',
    backgroundColor: '#0f172a',
    borderRadius: '6px',
    fontSize: '0.75rem',
    color: '#94a3b8',
    border: '1px solid #1e293b',
  },
};