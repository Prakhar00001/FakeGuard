import React, { useState } from 'react';

// Use relative API path so it works seamlessly locally & on Vercel
const API_ENDPOINT = '/api/predict';

export default function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!review.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_text: review }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('API Error:', err);
      setError('Failed to analyze review. Ensure backend API is active.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>🛡️ FakeGuard</h1>
        <p style={styles.subtitle}>
          Full-Stack Fake Review Detection Powered by Stacked Ensemble ML
        </p>
      </header>

      <main style={styles.main}>
        <form onSubmit={handleAnalyze} style={styles.form}>
          <label style={styles.label}>Paste Review Text:</label>
          <textarea
            rows="5"
            style={styles.textarea}
            placeholder="e.g., AMAZING PRODUCT MUST BUY NOW 100% BEST EVER!!!"
            value={review}
            onChange={(e) => setReview(e.target.value)}
          />
          <button
            type="submit"
            disabled={loading || !review.trim()}
            style={loading ? { ...styles.button, ...styles.buttonDisabled } : styles.button}
          >
            {loading ? 'Analyzing Review...' : 'Analyze Review'}
          </button>
        </form>

        {error && <div style={styles.errorBox}>{error}</div>}

        {result && (
          <div style={styles.resultCard}>
            <div
              style={{
                ...styles.badge,
                backgroundColor: result.is_fake ? '#fee2e2' : '#dcfce7',
                color: result.is_fake ? '#991b1b' : '#166534',
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

            <h3 style={styles.featureHeader}>15+ Behavioral & Linguistic Breakdown</h3>
            <div style={styles.featureGrid}>
              {Object.entries(result.linguistic_features || {}).map(([key, value]) => (
                <div key={key} style={styles.featurePill}>
                  <span style={styles.featureKey}>{key.replace('_', ' ')}:</span>{' '}
                  <strong style={styles.featureVal}>
                    {typeof value === 'number' ? value.toFixed(3) : value}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Inline Styles for Clean Dashboard UI
const styles = {
  container: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    maxWidth: '850px',
    margin: '0 auto',
    padding: '2rem 1rem',
    color: '#1f2937',
  },
  header: {
    textAlign: 'center',
    marginBottom: '2rem',
  },
  title: {
    fontSize: '2.25rem',
    fontWeight: '800',
    color: '#111827',
    margin: '0 0 0.5rem 0',
  },
  subtitle: {
    fontSize: '1rem',
    color: '#4b5563',
    margin: 0,
  },
  main: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    backgroundColor: '#ffffff',
    padding: '1.5rem',
    borderRadius: '12px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    border: '1px solid #e5e7eb',
  },
  label: {
    fontWeight: '600',
    fontSize: '0.95rem',
    color: '#374151',
  },
  textarea: {
    width: '100%',
    padding: '0.75rem',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
    fontSize: '1rem',
    boxSizing: 'border-box',
    outline: 'none',
    resize: 'vertical',
  },
  button: {
    padding: '0.75rem 1.5rem',
    backgroundColor: '#2563eb',
    color: '#ffffff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.2s ease',
  },
  buttonDisabled: {
    backgroundColor: '#93c5fd',
    cursor: 'not-allowed',
  },
  errorBox: {
    padding: '1rem',
    backgroundColor: '#fef2f2',
    color: '#991b1b',
    border: '1px solid #fecaca',
    borderRadius: '8px',
  },
  resultCard: {
    backgroundColor: '#ffffff',
    padding: '1.5rem',
    borderRadius: '12px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    border: '1px solid #e5e7eb',
  },
  badge: {
    display: 'inline-block',
    padding: '0.5rem 1rem',
    borderRadius: '9999px',
    fontWeight: '700',
    fontSize: '1rem',
    marginBottom: '1rem',
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
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    border: '1px solid #f3f4f6',
  },
  metricLabel: {
    fontSize: '0.85rem',
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  metricValue: {
    fontSize: '1.5rem',
    fontWeight: '700',
    color: '#111827',
  },
  featureHeader: {
    fontSize: '1.1rem',
    fontWeight: '700',
    marginBottom: '0.75rem',
    color: '#374151',
  },
  featureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: '0.5rem',
  },
  featurePill: {
    padding: '0.5rem 0.75rem',
    backgroundColor: '#f3f4f6',
    borderRadius: '6px',
    fontSize: '0.85rem',
    color: '#4b5563',
  },
  featureKey: {
    textTransform: 'capitalize',
  },
  featureVal: {
    color: '#111827',
  },
};