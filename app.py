import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load AdaBoost Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ada_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("AdaBoost model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Single-Page HTML + CSS + JS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg-gradient: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0f172a);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.125);
            --accent-glow: #6366f1;
        }

        body {
            background: var(--bg-gradient);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: #f8fafc;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 1.25rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.25);
        }

        .header-title {
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        .form-label {
            font-weight: 600;
            color: #cbd5e1;
            font-size: 0.9rem;
        }

        .form-control, .form-select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: #f8fafc;
            border-radius: 0.5rem;
        }

        .form-control:focus, .form-select:focus {
            background: rgba(15, 23, 42, 0.8);
            border-color: #6366f1;
            color: #fff;
            box-shadow: 0 0 0 0.25rem rgba(99, 102, 241, 0.25);
        }

        .range-badge {
            background: rgba(99, 102, 241, 0.2);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 0.375rem;
            padding: 0.2rem 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .btn-predict {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            border: none;
            color: white;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 0.8rem 1.5rem;
            border-radius: 0.75rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .btn-predict:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
            color: #fff;
        }

        /* Pulse Animation for Prediction Result */
        .pulse-animation {
            animation: pulseGlow 2s infinite;
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(99, 102, 241, 0); }
            100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }

        .result-box {
            display: none;
            border-radius: 1rem;
            padding: 1.5rem;
            text-align: center;
        }

        .chart-container {
            position: relative;
            min-height: 260px;
            width: 100%;
        }
    </style>
</head>
<body class="py-5">

<div class="container">
    <!-- Header -->
    <div class="row justify-content-center text-center mb-5">
        <div class="col-lg-8">
            <i class="fa-solid fa-brain fa-3x text-indigo mb-3" style="color: #818cf8;"></i>
            <h1 class="header-title display-4">AdaBoost Customer Intelligence</h1>
            <p class="text-secondary fs-5">Predict Customer Churn Risk & Analyze Feature Profile in Real-time</p>
        </div>
    </div>

    <div class="row g-4">
        <!-- Input Form Section -->
        <div class="col-lg-7">
            <div class="glass-card p-4 h-100">
                <h4 class="mb-4 text-light"><i class="fa-solid fa-sliders me-2 text-primary"></i> Customer Parameters</h4>
                <form id="predictionForm">
                    <div class="row g-3">
                        
                        <!-- Numerical Features -->
                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-user me-1"></i> Age</span>
                                <span class="range-badge" id="val_Age">35</span>
                            </label>
                            <input type="range" class="form-range" name="Age" min="18" max="100" value="35" oninput="updateVal('Age', this.value)">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label">Gender</label>
                            <select class="form-select" name="Gender">
                                <option value="0">Female</option>
                                <option value="1">Male</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-calendar-days me-1"></i> Tenure (Months)</span>
                                <span class="range-badge" id="val_Tenure">24</span>
                            </label>
                            <input type="range" class="form-range" name="Tenure" min="0" max="120" value="24" oninput="updateVal('Tenure', this.value)">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-chart-line me-1"></i> Usage Frequency</span>
                                <span class="range-badge" id="val_Usage Frequency">15</span>
                            </label>
                            <input type="range" class="form-range" name="Usage Frequency" min="1" max="50" value="15" oninput="updateVal('Usage Frequency', this.value)">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-headset me-1"></i> Support Calls</span>
                                <span class="range-badge" id="val_Support Calls">2</span>
                            </label>
                            <input type="range" class="form-range" name="Support Calls" min="0" max="20" value="2" oninput="updateVal('Support Calls', this.value)">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-clock me-1"></i> Payment Delay (Days)</span>
                                <span class="range-badge" id="val_Payment Delay">5</span>
                            </label>
                            <input type="range" class="form-range" name="Payment Delay" min="0" max="60" value="5" oninput="updateVal('Payment Delay', this.value)">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label">Subscription Type</label>
                            <select class="form-select" name="Subscription Type">
                                <option value="0">Basic</option>
                                <option value="1">Standard</option>
                                <option value="2">Premium</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label">Contract Length</label>
                            <select class="form-select" name="Contract Length">
                                <option value="0">Monthly</option>
                                <option value="1">Quarterly</option>
                                <option value="2">Annual</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-dollar-sign me-1"></i> Total Spend ($)</span>
                                <span class="range-badge" id="val_Total Spend">500</span>
                            </label>
                            <input type="range" class="form-range" name="Total Spend" min="100" max="5000" step="50" value="500" oninput="updateVal('Total Spend', this.value)">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="fa-solid fa-handshake me-1"></i> Last Interaction (Days)</span>
                                <span class="range-badge" id="val_Last Interaction">10</span>
                            </label>
                            <input type="range" class="form-range" name="Last Interaction" min="0" max="90" value="10" oninput="updateVal('Last Interaction', this.value)">
                        </div>

                    </div>

                    <div class="mt-4">
                        <button type="submit" class="btn btn-predict w-100">
                            <i class="fa-solid fa-bolt me-2"></i> Evaluate Customer Risk Profile
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <!-- Output & Visualization Section -->
        <div class="col-lg-5">
            <div class="glass-card p-4 h-100 d-flex flex-column justify-content-between">
                <div>
                    <h4 class="mb-4 text-light"><i class="fa-solid fa-chart-pie me-2 text-warning"></i> Predictive Analytics</h4>
                    
                    <!-- Prediction Result Alert -->
                    <div id="resultBox" class="result-box pulse-animation mb-4">
                        <i id="resultIcon" class="fa-solid fa-2x mb-2"></i>
                        <h3 id="resultTitle" class="fw-bold mb-1"></h3>
                        <p id="resultDesc" class="mb-0 small"></p>
                    </div>

                    <!-- Radar Visualization -->
                    <div class="chart-container mb-3">
                        <canvas id="featureChart"></canvas>
                    </div>
                </div>

                <div class="text-center text-muted small border-top border-secondary pt-3">
                    <i class="fa-solid fa-microchip me-1"></i> Powered by 50-Estimator AdaBoost Ensemble
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    function updateVal(id, val) {
        document.getElementById('val_' + id).innerText = val;
        updateChartFromInputs();
    }

    let chartInstance = null;

    function initChart() {
        const ctx = document.getElementById('featureChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Tenure', 'Usage Freq', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction'],
                datasets: [{
                    label: 'Normalized Profile',
                    data: [20, 30, 10, 8, 10, 11],
                    fill: true,
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: '#818cf8',
                    pointBackgroundColor: '#6366f1',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#6366f1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#cbd5e1', font: { size: 10 } },
                        ticks: { display: false, max: 100, min: 0 }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });
    }

    function updateChartFromInputs() {
        if (!chartInstance) return;
        const form = document.getElementById('predictionForm');
        const formData = new FormData(form);

        // Normalize features for visualization radar
        const tenure = (parseFloat(formData.get('Tenure')) / 120) * 100;
        const usage = (parseFloat(formData.get('Usage Frequency')) / 50) * 100;
        const calls = (parseFloat(formData.get('Support Calls')) / 20) * 100;
        const delay = (parseFloat(formData.get('Payment Delay')) / 60) * 100;
        const spend = (parseFloat(formData.get('Total Spend')) / 5000) * 100;
        const interaction = (parseFloat(formData.get('Last Interaction')) / 90) * 100;

        chartInstance.data.datasets[0].data = [tenure, usage, calls, delay, spend, interaction];
        chartInstance.update();
    }

    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => data[key] = parseFloat(value));

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        const resultBox = document.getElementById('resultBox');
        const resultIcon = document.getElementById('resultIcon');
        const resultTitle = document.getElementById('resultTitle');
        const resultDesc = document.getElementById('resultDesc');

        resultBox.style.display = 'block';

        if (result.prediction === 1) {
            resultBox.style.background = 'rgba(239, 68, 68, 0.2)';
            resultBox.style.border = '1px solid rgba(239, 68, 68, 0.5)';
            resultBox.style.color = '#fca5a5';
            resultIcon.className = 'fa-solid fa-triangle-exclamation fa-2x mb-2 text-danger';
            resultTitle.innerText = 'High Churn Risk Detected';
            resultDesc.innerText = 'Customer profile exhibits characteristics associated with subscription cancellation.';
        } else {
            resultBox.style.background = 'rgba(34, 197, 94, 0.2)';
            resultBox.style.border = '1px solid rgba(34, 197, 94, 0.5)';
            resultBox.style.color = '#86efac';
            resultIcon.className = 'fa-solid fa-circle-check fa-2x mb-2 text-success';
            resultTitle.innerText = 'Low Churn Risk';
            resultDesc.innerText = 'Customer retention probability is high. Customer is in good standing.';
        }
    });

    window.onload = function() {
        initChart();
        updateChartFromInputs();
    };
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not loaded.'}), 500

    try:
        data = request.get_json()
        
        # Exact feature ordering expected by the AdaBoost Model
        feature_order = [
            'Age', 'Gender', 'Tenure', 'Usage Frequency',
            'Support Calls', 'Payment Delay', 'Subscription Type',
            'Contract Length', 'Total Spend', 'Last Interaction'
        ]
        
        # Build feature vector in correct order
        features = [data[feature] for feature in feature_order]
        features_array = np.array([features])

        # Model prediction
        prediction = model.predict(features_array)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
