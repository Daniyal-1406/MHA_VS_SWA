import { useState } from 'react'

const CLASSES   = ['World', 'Sports', 'Business', 'Sci/Tech']
const CLASS_COLOR = {
  World:    { fill: '#3b82f6', bg: 'rgba(59,130,246,0.15)',  text: '#93c5fd' },
  Sports:   { fill: '#22c55e', bg: 'rgba(34,197,94,0.15)',   text: '#86efac' },
  Business: { fill: '#f59e0b', bg: 'rgba(245,158,11,0.15)',  text: '#fcd34d' },
  'Sci/Tech':{ fill: '#a855f7', bg: 'rgba(168,85,247,0.15)', text: '#d8b4fe' },
}

const EXAMPLES = [
  'NASA astronauts complete historic spacewalk to repair the International Space Station.',
  'Stock markets tumble as Federal Reserve signals another interest rate hike.',
  'Manchester United beats Chelsea in a thrilling Premier League showdown.',
  'OpenAI releases GPT-5 with multi-modal reasoning capabilities.',
]

export default function App() {
  const [text,    setText]    = useState('')
  const [model,   setModel]   = useState('mha')
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')

  async function predict() {
    if (!text.trim()) return
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const res  = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text: text.trim(), model }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      setResult(await res.json())
    } catch (e) {
      setError(e.message || 'Could not reach the API. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  function loadExample(ex) { setText(ex); setResult(null); setError('') }

  const palette = result ? CLASS_COLOR[result.predicted_class] : null

  return (
    <div className="page">

      {/* ── header ──────────────────────────────────────── */}
      <header className="header">
        <h1>AG News Classifier</h1>
        <p>Classify news into World · Sports · Business · Sci/Tech using custom Transformers</p>
      </header>

      {/* ── main card ────────────────────────────────────── */}
      <div className="card">

        {/* model toggle */}
        <div className="toggle-row">
          <span className="toggle-label">Attention model</span>
          <div className="toggle">
            <button className={model === 'mha' ? 'active' : ''} onClick={() => { setModel('mha'); setResult(null) }}>
              MHA
            </button>
            <button className={model === 'swa' ? 'active' : ''} onClick={() => { setModel('swa'); setResult(null) }}>
              SWA
            </button>
          </div>
        </div>

        {/* textarea */}
        <div className="input-area">
          <textarea
            value={text}
            onChange={e => { setText(e.target.value); setResult(null) }}
            placeholder="Paste or type a news headline or article…"
            onKeyDown={e => e.key === 'Enter' && e.ctrlKey && predict()}
          />
          <span className="char-count">{text.length} chars</span>
        </div>

        {/* example pills */}
        <div style={{ display:'flex', flexWrap:'wrap', gap:'8px', marginBottom:'20px' }}>
          {EXAMPLES.map((ex, i) => (
            <button key={i} onClick={() => loadExample(ex)} style={{
              background:'rgba(255,255,255,0.05)', border:'1px solid rgba(255,255,255,0.1)',
              borderRadius:'999px', padding:'5px 14px', color:'#94a3b8',
              fontSize:'0.75rem', cursor:'pointer', fontFamily:'inherit',
              transition:'all 0.15s'
            }}
            onMouseEnter={e => e.target.style.color='#e2e8f0'}
            onMouseLeave={e => e.target.style.color='#94a3b8'}
            >
              Example {i + 1}
            </button>
          ))}
        </div>

        {/* predict button */}
        <button className="btn-predict" onClick={predict} disabled={loading || !text.trim()}>
          {loading ? <><span className="spinner"/>Classifying…</> : 'Classify →'}
        </button>

        {/* error */}
        {error && <div className="error">{error}</div>}

        {/* result */}
        {result && (
          <div className="result" style={{ borderColor: palette.fill + '40' }}>

            <div className="result-top">
              <span className="result-badge" style={{ background: palette.bg, color: palette.text }}>
                {result.predicted_class}
              </span>
              <span className="result-class" style={{ color: palette.text }}>
                {result.predicted_class}
              </span>
              <span className="model-tag">{result.model}</span>
            </div>

            <div className="prob-section">
              <h4>Confidence Breakdown</h4>
              {CLASSES.map(cls => {
                const pct = (result.probabilities[cls] * 100).toFixed(1)
                const col = CLASS_COLOR[cls]
                return (
                  <div className="prob-row" key={cls}>
                    <span className="prob-name">{cls}</span>
                    <div className="prob-track">
                      <div className="prob-fill" style={{ width: `${pct}%`, background: col.fill }} />
                    </div>
                    <span className="prob-pct">{pct}%</span>
                  </div>
                )
              })}
            </div>

          </div>
        )}
      </div>

      <p className="footer">
        MHA · SWA from scratch in PyTorch &nbsp;·&nbsp; AG News (4-class) &nbsp;·&nbsp;
        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
           style={{ color:'#6366f1', textDecoration:'none' }}>API Docs ↗</a>
      </p>
    </div>
  )
}
