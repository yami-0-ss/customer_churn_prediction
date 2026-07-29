import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load the AdaBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ada_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found. Ensure the model file is uploaded.")

# Feature details based on the serialized model meta
FEATURES = [
    {"name": "Age", "type": "number", "default": 35, "min": 18, "max": 100, "step": 1},
    {"name": "Gender", "type": "select", "options": [("0", "Female"), ("1", "Male")]},
    {"name": "Tenure", "type": "number", "default": 12, "min": 0, "max": 120, "step": 1},
    {"name": "Usage Frequency", "type": "number", "default": 15, "min": 0, "max": 100, "step": 1},
    {"name": "Support Calls", "type": "number", "default": 2, "min": 0, "max": 50, "step": 1},
    {"name": "Payment Delay", "type": "number", "default": 1, "min": 0, "max": 30, "step": 1},
    {"name": "Subscription Type", "type": "select", "options": [("0", "Basic"), ("1", "Standard"), ("2", "Premium")]},
    {"name": "Contract Length", "type": "select", "options": [("0", "Monthly"), ("1", "Quarterly"), ("2", "Annual")]},
    {"name": "Total Spend", "type": "number", "default": 500.0, "min": 0, "max": 10000, "step": 10},
    {"name": "Last Interaction", "type": "number", "default": 5, "min": 0, "max": 365, "step": 1},
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Predictive Analytics Dashboard</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(23, 32, 54, 0.65);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-green: #10b981;
            --accent-red: #f43f5e;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Animated Glowing Orbs Background */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.35;
            z-index: 0;
            animation: float 12s infinite alternate ease-in-out;
        }

        .orb-1 {
            width: 350px;
            height: 350px;
            background: #6366f1;
            top: -50px;
            left: -50px;
        }

        .orb-2 {
            width: 400px;
            height: 400px;
            background: #a855f7;
            bottom: -80px;
            right: -80px;
            animation-delay: -6s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(40px, 50px) scale(1.1); }
        }

        /* Container */
        .container {
            width: 100%;
            max-width: 1100px;
            z-index: 1;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            padding: 2.5rem;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Layout Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 868px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Form Inputs */
        .form-section {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        .btn-submit {
            grid-column: span 2;
            margin-top: 0.5rem;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.9rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Visualization Section */
        .viz-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            position: relative;
        }

        .result-card {
            text-align: center;
            margin-bottom: 1.5rem;
            width: 100%;
        }

        .result-title {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 700;
            transition: color 0.3s ease;
        }

        .chart-container {
            position: relative;
            width: 100%;
            max-width: 280px;
            height: 280px;
        }

        /* Status Badge */
        .badge {
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }

        .badge-positive {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-negative {
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <header>
            <h1>AdaBoost Predictive Intelligence</h1>
            <p>Input model features to compute live predictions and confidence breakdown.</p>
        </header>

        <div class="dashboard-grid">
            <!-- Form Section -->
            <form id="predictForm" class="form-section">
                <div class="form-grid">
                    {% for feat in features %}
                    <div class="input-group">
                        <label for="{{ feat.name }}">{{ feat.name }}</label>
                        {% if feat.type == 'select' %}
                        <select id="{{ feat.name }}" name="{{ feat.name }}">
                            {% for val, label in feat.options %}
                            <option value="{{ val }}">{{ label }}</option>
                            {% endfor %}
                        </select>
                        {% else %}
                        <input type="number" id="{{ feat.name }}" name="{{ feat.name }}" 
                               value="{{ feat.default }}" min="{{ feat.min }}" max="{{ feat.max }}" step="{{ feat.step }}" required>
                        {% endif %}
                    </div>
                    {% endfor %}
                    <button type="submit" class="btn-submit">Run Prediction</button>
                </div>
            </form>

            <!-- Visualization Section -->
            <div class="viz-section">
                <div class="result-card">
                    <div class="result-title">Predicted Class</div>
                    <div id="resultValue" class="result-value">--</div>
                    <div id="resultBadge" class="badge" style="display:none;"></div>
                </div>

                <div class="chart-container">
                    <canvas id="probabilityChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chartInstance = null;

        // Initialize Chart.js Donut Chart
        function initChart(prob0 = 50, prob1 = 50) {
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            
            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Class 0', 'Class 1'],
                    datasets: [{
                        data: [prob0, prob1],
                        backgroundColor: ['rgba(99, 102, 241, 0.8)', 'rgba(168, 85, 247, 0.8)'],
                        borderColor: ['#6366f1', '#a855f7'],
                        borderWidth: 2,
                        hoverOffset: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans' } }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return ' ' + context.label + ': ' + context.raw + '%';
                                }
                            }
                        }
                    },
                    cutout: '70%',
                    animation: {
                        animateScale: true,
                        animateRotate: true,
                        duration: 1000
                    }
                }
            });
        }

        // Handle AJAX submission
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => data[key] = value);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.error) {
                    alert('Error: ' + result.error);
                    return;
                }

                // Update UI elements
                const resValue = document.getElementById('resultValue');
                const resBadge = document.getElementById('resultBadge');

                resValue.textContent = 'Class ' + result.prediction;
                resBadge.style.display = 'inline-block';
                
                if (result.prediction === 1) {
                    resValue.style.color = '#a855f7';
                    resBadge.className = 'badge badge-positive';
                    resBadge.textContent = 'High Confidence';
                } else {
                    resValue.style.color = '#6366f1';
                    resBadge.className = 'badge badge-negative';
                    resBadge.textContent = 'Standard Confidence';
                }

                // Update chart
                const prob0 = (result.probabilities[0] * 100).toFixed(1);
                const prob1 = (result.probabilities[1] * 100).toFixed(1);
                initChart(prob0, prob1);

            } catch (err) {
                console.error(err);
                alert('Prediction failed. Ensure server is running.');
            }
        });

        // Initial Chart Render
        initChart(50, 50);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found on server."}), 500

    try:
        data = request.get_json()
        
        # Extract features in exact order expected by the model
        feature_values = [
            float(data.get("Age", 0)),
            float(data.get("Gender", 0)),
            float(data.get("Tenure", 0)),
            float(data.get("Usage Frequency", 0)),
            float(data.get("Support Calls", 0)),
            float(data.get("Payment Delay", 0)),
            float(data.get("Subscription Type", 0)),
            float(data.get("Contract Length", 0)),
            float(data.get("Total Spend", 0)),
            float(data.get("Last Interaction", 0))
        ]

        features_array = np.array([feature_values])
        
        prediction = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0].tolist()

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
