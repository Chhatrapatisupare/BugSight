import gradio as gr
import joblib
import pandas as pd
from datetime import datetime

# ---------- Load Model ----------
pipe = joblib.load("models/bug_classifier_pipeline.pkl")

# ---------- Category metadata (for pretty UI) ----------
CATEGORY_META = {
    "UI":          {"icon": "🎨", "color": "#8b5cf6", "desc": "Frontend / design issues"},
    "Backend":     {"icon": "⚙️", "color": "#3b82f6", "desc": "Server / API / logic bugs"},
    "Database":    {"icon": "🗄️", "color": "#10b981", "desc": "Queries, constraints, data"},
    "Security":    {"icon": "🔒", "color": "#ef4444", "desc": "Vulnerabilities, exploits"},
    "Performance": {"icon": "⚡", "color": "#f59e0b", "desc": "Speed, memory, latency"},
}

# ---------- Core logic ----------
def classify(summary, description):
    text = f"{summary} {description}".strip()
    if not text:
        return "<div style='padding:20px;color:#ef4444;'>⚠️ Please enter bug text</div>"

    probs = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    pairs = sorted(zip(classes, probs), key=lambda x: -x[1])
    top_cat, top_conf = pairs[0]
    meta = CATEGORY_META.get(top_cat, {"icon": "🐞", "color": "#64748b"})

    # Build bar chart
    bars_html = ""
    for cat, prob in pairs:
        m = CATEGORY_META.get(cat, {"icon": "🐞", "color": "#64748b"})
        pct = prob * 100
        bars_html += f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:13px;margin-bottom:6px;color:#334155;">
                <span style="font-weight:600;">{m['icon']} {cat}</span>
                <span style="color:#64748b;">{pct:.1f}%</span>
            </div>
            <div style="background:#f1f5f9;border-radius:8px;height:10px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;
                            background:linear-gradient(90deg,{m['color']},{m['color']}cc);
                            border-radius:8px;transition:width 0.5s ease;"></div>
            </div>
        </div>
        """

    result_html = f"""
    <div style="font-family:'Inter','Segoe UI',sans-serif;">
        <div style="background:linear-gradient(135deg,{meta['color']}15,{meta['color']}05);
                    border:1px solid {meta['color']}30;border-radius:16px;
                    padding:20px;margin-bottom:20px;">
            <div style="display:flex;align-items:center;gap:14px;">
                <div style="font-size:44px;">{meta['icon']}</div>
                <div>
                    <div style="font-size:12px;color:#64748b;text-transform:uppercase;
                                letter-spacing:1px;font-weight:600;">Predicted Category</div>
                    <div style="font-size:26px;font-weight:800;color:{meta['color']};
                                margin-top:2px;">{top_cat}</div>
                    <div style="font-size:13px;color:#64748b;margin-top:2px;">
                        {top_conf*100:.1f}% confidence
                    </div>
                </div>
            </div>
        </div>
        <div style="font-size:13px;font-weight:600;color:#334155;
                    margin-bottom:12px;text-transform:uppercase;letter-spacing:0.5px;">
            All Categories
        </div>
        {bars_html}
    </div>
    """
    return result_html


def batch_classify(file):
    if file is None:
        return pd.DataFrame()
    df = pd.read_csv(file.name)
    df["summary"] = df["summary"].astype(str).replace("nan", "")
    df["description"] = df["description"].astype(str).replace("nan", "")
    df["text"] = df["summary"] + " " + df["description"]
    df["predicted_category"] = pipe.predict(df["text"])
    df["confidence"] = pipe.predict_proba(df["text"]).max(axis=1).round(3)
    # Reorder
    return df[["summary", "description", "predicted_category", "confidence"]]


# ---------- Custom CSS ----------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif !important; }

body, .gradio-container {
    background: #f8fafc !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Hide Gradio footer */
footer { display: none !important; }
.gradio-container > .main { padding: 0 !important; }

/* ================= SIDEBAR ================= */
#sidebar {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    min-height: 100vh;
    padding: 28px 20px !important;
    color: white !important;
    border-right: 1px solid #1e293b;
}
#sidebar .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
    color: white;
}
#sidebar .tagline {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 32px;
    line-height: 1.5;
}
#sidebar .nav-section {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin: 24px 0 10px;
}
#sidebar .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 10px;
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
}
#sidebar .nav-item:hover {
    background: rgba(255,255,255,0.06);
    color: white;
}
#sidebar .nav-item.active {
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    color: white;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35);
}
#sidebar .badge {
    background: rgba(59,130,246,0.15);
    color: #60a5fa;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
    margin-left: auto;
    letter-spacing: 0.5px;
}

/* ================= MAIN AREA ================= */
#main-area {
    padding: 32px 40px !important;
    background: #f8fafc !important;
    min-height: 100vh;
}
#topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid #e2e8f0;
}
#topbar .title {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
}
#topbar .subtitle {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}
#topbar .status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #ecfdf5;
    color: #059669;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
#topbar .dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* ================= CARDS ================= */
.panel {
    background: white !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04) !important;
    border: 1px solid #e2e8f0 !important;
    height: 100%;
}
.panel-title {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 0 16px 0 !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ================= FORM ELEMENTS ================= */
label, .label-wrap span, span[data-testid="block-info"] {
    background: transparent !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
input, textarea, .gr-text-input, .gr-textarea {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
input:focus, textarea:focus {
    border-color: #3b82f6 !important;
    background: white !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.1) !important;
}

/* Primary Button */
button.primary, .gr-button-primary {
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(59,130,246,0.45) !important;
}

/* Secondary Button */
button.secondary {
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
}
button.secondary:hover {
    background: #e2e8f0 !important;
}

/* ================= TABS (still used internally) ================= */
.tab-nav { border-bottom: 2px solid #e2e8f0 !important; margin-bottom: 24px !important; }
.tab-nav button {
    font-weight: 600 !important;
    color: #64748b !important;
    border: none !important;
    padding: 12px 20px !important;
    font-size: 14px !important;
}
.tab-nav button.selected {
    color: #3b82f6 !important;
    border-bottom: 3px solid #3b82f6 !important;
    background: transparent !important;
}

/* ================= METRIC CARDS ================= */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.metric-card .label {
    font-size: 12px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 6px;
}
.metric-card .change {
    font-size: 12px;
    color: #10b981;
    font-weight: 600;
    margin-top: 4px;
}

/* ================= EXAMPLES ================= */
.examples-table, .examples-table table {
    background: white !important;
    border-radius: 10px !important;
}
.examples-table td {
    padding: 10px 14px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    font-size: 13px !important;
    color: #334155 !important;
    cursor: pointer !important;
}
.examples-table tr:hover td {
    background: #f8fafc !important;
}

/* Fix label output */
.gr-label {
    background: transparent !important;
}
"""

# ---------- UI ----------
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
    title="BugSight · Bug Report Classifier",
) as demo:

    with gr.Row(equal_height=False):

        # ================= SIDEBAR =================
        with gr.Column(scale=1, elem_id="sidebar", min_width=260):
            gr.HTML("""
            <div class="logo">🐞 BugSight</div>
            <div class="tagline">Automated Bug Report<br>Classification System</div>

            <div class="nav-section">Workspace</div>
            <a class="nav-item active">🔍 <span>Classify Bug</span></a>
            <a class="nav-item">📂 <span>Batch Process</span></a>
            <a class="nav-item">📊 <span>Analytics</span></a>

            <div class="nav-section">System</div>
            <a class="nav-item">🧠 <span>Model Info</span> <span class="badge">LR</span></a>
            <a class="nav-item">ℹ️ <span>About</span></a>

            <div style="margin-top:40px;padding:14px;background:rgba(59,130,246,0.1);
                        border-radius:12px;border:1px solid rgba(59,130,246,0.2);">
                <div style="font-size:11px;color:#60a5fa;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;">Model Status</div>
                <div style="font-size:13px;color:#e2e8f0;margin-top:6px;font-weight:600;">
                    ✅ Trained & Ready
                </div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px;">
                    15 samples · 5 categories
                </div>
            </div>
            """)

        # ================= MAIN AREA =================
        with gr.Column(scale=4, elem_id="main-area"):

            gr.HTML(f"""
            <div id="topbar">
                <div>
                    <h1 class="title">Bug Report Classification</h1>
                    <div class="subtitle">Powered by TF-IDF + Logistic Regression · NLP Pipeline</div>
                </div>
                <div class="status">
                    <span class="dot"></span>
                    <span>System Online</span>
                </div>
            </div>
            """)

            with gr.Tabs():

                # ---------- TAB 1: SINGLE ----------
                with gr.Tab("🔍 Classify Bug"):
                    with gr.Row(equal_height=False):

                        with gr.Column(scale=1, elem_classes="panel"):
                            gr.HTML('<div class="panel-title">📝 Bug Details</div>')
                            summary = gr.Textbox(
                                label="Summary",
                                placeholder="Login button not working...",
                                lines=1,
                            )
                            desc = gr.Textbox(
                                label="Description",
                                placeholder="Provide full details of the bug...",
                                lines=7,
                            )
                            with gr.Row():
                                btn = gr.Button("🚀 Classify", variant="primary", scale=3)
                                clear = gr.Button("Clear", variant="secondary", scale=1)

                        with gr.Column(scale=1, elem_classes="panel"):
                            gr.HTML('<div class="panel-title">🎯 Prediction</div>')
                            out = gr.HTML(
                                "<div style='color:#94a3b8;text-align:center;"
                                "padding:60px 20px;font-size:14px;'>"
                                "Enter a bug report and click <b>Classify</b> "
                                "to see results</div>"
                            )

                    gr.HTML('<div style="height:8px;"></div>')
                    with gr.Column(elem_classes="panel"):
                        gr.HTML('<div class="panel-title">💡 Quick Examples</div>')
                        gr.Examples(
                            examples=[
                                ["Login button not working", "Clicking login does nothing on Chrome."],
                                ["SQL injection in search", "Unsanitized input allows SQL payloads."],
                                ["Database timeout", "Query takes too long to respond."],
                                ["App crashes on startup", "Crash after splash screen on Android 13."],
                                ["Page loads very slowly", "Dashboard takes 12 seconds to load."],
                            ],
                            inputs=[summary, desc],
                            label="",
                        )

                    btn.click(classify, [summary, desc], out)
                    clear.click(lambda: ("", "", "<div style='color:#94a3b8;text-align:center;"
                                                  "padding:60px 20px;font-size:14px;'>"
                                                  "Enter a bug report and click <b>Classify</b> "
                                                  "to see results</div>"),
                                None, [summary, desc, out])

                # ---------- TAB 2: BATCH ----------
                with gr.Tab("📂 Batch Process"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, elem_classes="panel"):
                            gr.HTML('<div class="panel-title">📁 Upload CSV</div>')
                            gr.HTML(
                                "<div style='font-size:13px;color:#64748b;"
                                "margin-bottom:16px;line-height:1.6;'>"
                                "CSV must have columns: <code style='background:#f1f5f9;"
                                "padding:2px 6px;border-radius:4px;'>summary</code>, "
                                "<code style='background:#f1f5f9;padding:2px 6px;"
                                "border-radius:4px;'>description</code></div>"
                            )
                            file = gr.File(label="", file_types=[".csv"])

                        with gr.Column(scale=2, elem_classes="panel"):
                            gr.HTML('<div class="panel-title">📊 Results</div>')
                            table = gr.Dataframe(
                                label="",
                                wrap=True,
                                interactive=False,
                            )
                    file.change(batch_classify, file, table)

                # ---------- TAB 3: ANALYTICS ----------
                with gr.Tab("📊 Analytics"):
                    gr.HTML("""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="label">Total Categories</div>
                            <div class="value">5</div>
                            <div class="change">↑ UI · Backend · DB · Sec · Perf</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Training Samples</div>
                            <div class="value">15</div>
                            <div class="change">↑ Synthetic + real bugs</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Algorithm</div>
                            <div class="value" style="font-size:22px;">Logistic</div>
                            <div class="change">+ TF-IDF Vectorizer</div>
                        </div>
                    </div>
                    """)

                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes="panel"):
                            gr.HTML('<div class="panel-title">📈 Category Distribution</div>')
                            gr.HTML("""
                            <div style="padding:10px 0;">
                                <div style="display:flex;justify-content:space-between;
                                            margin-bottom:8px;font-size:13px;color:#475569;">
                                    <span>🎨 UI</span><span style="font-weight:600;">3 samples</span>
                                </div>
                                <div style="background:#f1f5f9;height:8px;border-radius:4px;
                                            margin-bottom:16px;overflow:hidden;">
                                    <div style="width:100%;height:100%;
                                                background:linear-gradient(90deg,#8b5cf6,#a78bfa);
                                                border-radius:4px;"></div>
                                </div>

                                <div style="display:flex;justify-content:space-between;
                                            margin-bottom:8px;font-size:13px;color:#475569;">
                                    <span>⚙️ Backend</span><span style="font-weight:600;">3 samples</span>
                                </div>
                                <div style="background:#f1f5f9;height:8px;border-radius:4px;
                                            margin-bottom:16px;overflow:hidden;">
                                    <div style="width:100%;height:100%;
                                                background:linear-gradient(90deg,#3b82f6,#60a5fa);
                                                border-radius:4px;"></div>
                                </div>

                                <div style="display:flex;justify-content:space-between;
                                            margin-bottom:8px;font-size:13px;color:#475569;">
                                    <span>🗄️ Database</span><span style="font-weight:600;">3 samples</span>
                                </div>
                                <div style="background:#f1f5f9;height:8px;border-radius:4px;
                                            margin-bottom:16px;overflow:hidden;">
                                    <div style="width:100%;height:100%;
                                                background:linear-gradient(90deg,#10b981,#34d399);
                                                border-radius:4px;"></div>
                                </div>

                                <div style="display:flex;justify-content:space-between;
                                            margin-bottom:8px;font-size:13px;color:#475569;">
                                    <span>🔒 Security</span><span style="font-weight:600;">3 samples</span>
                                </div>
                                <div style="background:#f1f5f9;height:8px;border-radius:4px;
                                            margin-bottom:16px;overflow:hidden;">
                                    <div style="width:100%;height:100%;
                                                background:linear-gradient(90deg,#ef4444,#f87171);
                                                border-radius:4px;"></div>
                                </div>

                                <div style="display:flex;justify-content:space-between;
                                            margin-bottom:8px;font-size:13px;color:#475569;">
                                    <span>⚡ Performance</span><span style="font-weight:600;">3 samples</span>
                                </div>
                                <div style="background:#f1f5f9;height:8px;border-radius:4px;
                                            overflow:hidden;">
                                    <div style="width:100%;height:100%;
                                                background:linear-gradient(90deg,#f59e0b,#fbbf24);
                                                border-radius:4px;"></div>
                                </div>
                            </div>
                            """)

                        with gr.Column(elem_classes="panel"):
                            gr.HTML('<div class="panel-title">🧠 Model Pipeline</div>')
                            gr.HTML("""
                            <div style="font-size:13px;line-height:2;color:#475569;">
                                <div style="padding:10px;background:#f8fafc;border-radius:8px;
                                            margin-bottom:8px;">
                                    <b>1. Input Text</b><br>
                                    <span style="color:#64748b;">Summary + Description merged</span>
                                </div>
                                <div style="padding:10px;background:#f8fafc;border-radius:8px;
                                            margin-bottom:8px;">
                                    <b>2. TF-IDF Vectorizer</b><br>
                                    <span style="color:#64748b;">Unigrams + Bigrams, top 10k features</span>
                                </div>
                                <div style="padding:10px;background:#f8fafc;border-radius:8px;
                                            margin-bottom:8px;">
                                    <b>3. Logistic Regression</b><br>
                                    <span style="color:#64748b;">Balanced class weights, 1000 iterations</span>
                                </div>
                                <div style="padding:10px;background:#f8fafc;border-radius:8px;">
                                    <b>4. Softmax Output</b><br>
                                    <span style="color:#64748b;">5-class probability distribution</span>
                                </div>
                            </div>
                            """)

                # ---------- TAB 4: MODEL INFO ----------
                with gr.Tab("🧠 Model Info"):
                    with gr.Column(elem_classes="panel"):
                        gr.HTML('<div class="panel-title">🔬 Technical Specification</div>')
                        gr.HTML("""
                        <table style="width:100%;font-size:14px;color:#334155;
                                      border-collapse:collapse;">
                            <tr style="border-bottom:1px solid #f1f5f9;">
                                <td style="padding:12px 8px;color:#64748b;width:40%;">
                                    <b>Model Type</b></td>
                                <td style="padding:12px 8px;">Logistic Regression (multiclass)</td>
                            </tr>
                            <tr style="border-bottom:1px solid #f1f5f9;">
                                <td style="padding:12px 8px;color:#64748b;">
                                    <b>Vectorizer</b></td>
                                <td style="padding:12px 8px;">TF-IDF (1,2)-gram, max_features=10000</td>
                            </tr>
                            <tr style="border-bottom:1px solid #f1f5f9;">
                                <td style="padding:12px 8px;color:#64748b;">
                                    <b>Solver</b></td>
                                <td style="padding:12px 8px;">lbfgs</td>
                            </tr>
                            <tr style="border-bottom:1px solid #f1f5f9;">
                                <td style="padding:12px 8px;color:#64748b;">
                                    <b>Class Weight</b></td>
                                <td style="padding:12px 8px;">Balanced (handles imbalance)</td>
                            </tr>
                            <tr style="border-bottom:1px solid #f1f5f9;">
                                <td style="padding:12px 8px;color:#64748b;">
                                    <b>Max Iterations</b></td>
                                <td style="padding:12px 8px;">1000</td>
                            </tr>
                            <tr style="border-bottom:1px solid #f1f5f9;">
                                <td style="padding:12px 8px;color:#64748b;">
                                    <b>Categories</b></td>
                                <td style="padding:12px 8px;">
                                    🎨 UI · ⚙️ Backend · 🗄️ Database · 🔒 Security · ⚡ Performance
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:12px 8px;color:#64748b;">
                                    <b>Saved Model</b></td>
                                <td style="padding:12px 8px;">
                                    <code style="background:#f1f5f9;padding:3px 8px;
                                                 border-radius:6px;font-size:12px;">
                                    models/bug_classifier_pipeline.pkl</code>
                                </td>
                            </tr>
                        </table>
                        """)

                # ---------- TAB 5: ABOUT ----------
                with gr.Tab("ℹ️ About"):
                    with gr.Column(elem_classes="panel"):
                        gr.HTML('<div class="panel-title">🐞 About BugSight</div>')
                        gr.HTML("""
                        <div style="font-size:14px;line-height:1.8;color:#334155;">
                            <p><b>BugSight</b> is an automated bug report classification system
                            that uses Natural Language Processing and Machine Learning to
                            categorize software bugs into predefined engineering categories.</p>

                            <h4 style="margin-top:24px;color:#0f172a;">🎯 Purpose</h4>
                            <p>Manual bug triage is slow and error-prone. BugSight automates this
                            by instantly predicting which team should handle a bug, reducing
                            triage time from minutes to milliseconds.</p>

                            <h4 style="margin-top:24px;color:#0f172a;">🛠️ Tech Stack</h4>
                            <ul style="line-height:2;">
                                <li><b>Python 3</b> — core language</li>
                                <li><b>Scikit-learn</b> — TF-IDF + Logistic Regression</li>
                                <li><b>Gradio</b> — web GUI framework</li>
                                <li><b>Pandas / NumPy</b> — data handling</li>
                                <li><b>Joblib</b> — model persistence</li>
                            </ul>

                            <h4 style="margin-top:24px;color:#0f172a;">🚀 Future Scope</h4>
                            <ul style="line-height:2;">
                                <li>BERT / Transformer-based classification</li>
                                <li>Severity + priority prediction</li>
                                <li>Duplicate bug report detection</li>
                                <li>Jira / GitHub Issues integration</li>
                                <li>Active learning feedback loop</li>
                            </ul>
                        </div>
                        """)

# ---------- Launch ----------
demo.launch()