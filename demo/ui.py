from __future__ import annotations

# Custom CSS for the Gradio Interface
CSS: str = """
/* Body & container dark theme defaults */
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #e2e8f0 !important;
}

/* Center main wrapper */
.main-container {
    max-width: 1024px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
}

/* Title & Header styles */
.title-header {
    text-align: center !important;
    margin-bottom: 2rem !important;
}
.title-header h1 {
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: -0.04em !important;
}
.title-header h2 {
    font-size: 1.25rem !important;
    font-weight: 500 !important;
    color: #10b981 !important; /* Emerald Green */
    margin: 0.25rem 0 0 0 !important;
    letter-spacing: 0.05em !important;
}
.title-header p {
    font-size: 0.8rem !important;
    color: #6b7280 !important; /* Slate Gray */
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    margin: 0.5rem 0 0 0 !important;
}

/* Summary Card Grid */
.metrics-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 1.25rem !important;
    margin-bottom: 1.5rem !important;
}
.metric-card {
    background-color: #111827 !important; /* Dark Gray */
    border: 1px solid #1f2937 !important;
    border-radius: 0.75rem !important;
    padding: 1.25rem !important;
    text-align: center !important;
    transition: all 0.2s ease-in-out !important;
}
.metric-card:hover {
    border-color: #10b981 !important; /* Highlight green on hover */
}
.metric-title {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #9ca3af !important;
    text-transform: uppercase !important;
    letter-spacing: 0.075em !important;
    margin-bottom: 0.35rem !important;
}
.metric-value {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #10b981 !important;
}

/* Confidence Table styles */
.table-container {
    border-radius: 0.5rem !important;
    overflow: hidden !important;
    border: 1px solid #1f2937 !important;
    margin-top: 0.5rem !important;
}
.confidence-table {
    width: 100% !important;
    border-collapse: collapse !important;
    text-align: left !important;
}
.confidence-table th {
    background-color: #111827 !important;
    color: #9ca3af !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.85rem !important;
    border-bottom: 1px solid #1f2937 !important;
}
.confidence-table td {
    padding: 0.6rem 1rem !important;
    font-size: 0.85rem !important;
    color: #e2e8f0 !important;
    border-bottom: 1px solid #1f2937 !important;
    background-color: #111827 !important;
}
.confidence-table tr:last-child td {
    border-bottom: none !important;
}

/* Message styling */
.status-msg {
    text-align: center !important;
    padding: 1.25rem !important;
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 0.5rem !important;
    color: #9ca3af !important;
    font-size: 0.9rem !important;
}
.error-box {
    padding: 1rem !important;
    background-color: #7f1d1d !important;
    border: 1px solid #ef4444 !important;
    border-radius: 0.5rem !important;
    color: #fca5a5 !important;
    margin-bottom: 1rem !important;
    font-size: 0.9rem !important;
}
"""

def generate_metrics_html(drone_count: int, time_ms: float, device: str) -> str:
    """Generates three clean metric cards in HTML format."""
    return f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Drone Count</div>
            <div class="metric-value">{drone_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Inference Time</div>
            <div class="metric-value">{time_ms:.1f} ms</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Execution Device</div>
            <div class="metric-value">{device}</div>
        </div>
    </div>
    """

def generate_status_msg_html(text: str, is_error: bool = False) -> str:
    """Generates simple status or error boxes."""
    if is_error:
        return f'<div class="error-box">⚠️ {text}</div>'
    return f'<div class="status-msg">{text}</div>'
