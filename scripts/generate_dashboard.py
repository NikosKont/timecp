import json
import pathlib


def main():
    json_path = pathlib.Path('results/cp_calibration_data.json')
    if not json_path.exists():
        print(
            f'Error: JSON data file {json_path} does not exist. Run parse_summaries.py first.'
        )
        return

    # Load data
    with open(json_path) as f:
        data = json.load(f)

    # HTML/JS Dashboard Template
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conformal Prediction (CP) Calibration Dashboard</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>

    <style>
        :root {
            --bg-base: #080B10;
            --bg-surface: #0E131F;
            --bg-card: #151C2C;
            --bg-card-hover: #1A2338;
            --border: #222D44;
            --border-hover: #314264;

            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
            --text-muted: #6B7280;

            --color-purple: #8B5CF6;
            --color-purple-glow: rgba(139, 92, 246, 0.15);
            --color-blue: #3B82F6;
            --color-blue-glow: rgba(59, 130, 246, 0.15);
            --color-green: #10B981;
            --color-green-glow: rgba(16, 185, 129, 0.15);
            --color-pink: #EC4899;
            --color-pink-glow: rgba(236, 72, 153, 0.15);

            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 0.95rem;
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            overflow-x: hidden;
        }

        /* Sidebar Styling */
        aside {
            width: 320px;
            background-color: var(--bg-surface);
            border-right: 1px solid var(--border);
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            flex-shrink: 0;
            height: 100vh;
            position: sticky;
            top: 0;
            overflow-y: auto;
        }

        .sidebar-header {
            margin-bottom: 0.5rem;
        }

        .sidebar-header h2 {
            font-size: 1.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFF, var(--color-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .sidebar-header p {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }

        .filter-section {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .filter-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        select, input[type="text"] {
            width: 100%;
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        select:focus, input[type="text"]:focus {
            border-color: var(--color-purple);
            box-shadow: 0 0 0 3px var(--color-purple-glow);
        }

        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-height: 160px;
            overflow-y: auto;
            padding-right: 0.5rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            background-color: var(--bg-card);
        }

        /* Custom Scrollbar for group */
        .checkbox-group::-webkit-scrollbar {
            width: 4px;
        }
        .checkbox-group::-webkit-scrollbar-track {
            background: transparent;
        }
        .checkbox-group::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 2px;
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            font-size: 0.85rem;
            color: var(--text-secondary);
            transition: color 0.15s;
        }

        .checkbox-item:hover {
            color: var(--text-primary);
        }

        .checkbox-item input {
            cursor: pointer;
            accent-color: var(--color-purple);
        }

        /* Main Workspace */
        main {
            flex-grow: 1;
            padding: 2.5rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            overflow-y: auto;
            max-width: calc(100vw - 320px);
            min-width: 0;
        }

        /* KPI Row */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
        }

        .kpi-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-hover);
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: var(--accent-color, var(--color-purple));
        }

        .kpi-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .kpi-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* Navigation Tabs */
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
            gap: 1rem;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-sans);
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.75rem 1rem;
            cursor: pointer;
            transition: color 0.2s, border-color 0.2s;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }

        .tab-btn:hover {
            color: var(--text-primary);
        }

        .tab-btn.active {
            color: var(--color-purple);
            border-color: var(--color-purple);
        }

        .tab-content {
            display: none;
            flex-direction: column;
            gap: 2rem;
        }

        .tab-content.active {
            display: flex;
        }

        /* Layout Grids */
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1.5rem;
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            position: relative;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-title span.badge {
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background-color: var(--color-purple-glow);
            color: var(--color-purple);
            border: 1px solid var(--color-purple);
            font-weight: 600;
        }

        .chart-container {
            width: 100%;
            height: 400px;
            border-radius: 8px;
            overflow: hidden;
            background-color: #0E131F;
        }

        /* Tables */
        .table-wrapper {
            width: 100%;
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }

        th {
            background-color: var(--bg-surface);
            color: var(--text-primary);
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }

        td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
            font-family: monospace;
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background-color: var(--bg-card-hover);
            color: var(--text-primary);
            cursor: pointer;
        }

        .coverage-good {
            color: #10B981;
            font-weight: 600;
        }

        .coverage-bad {
            color: #EF4444;
            font-weight: 600;
        }

        /* Search / Table Controls */
        .table-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }

        .search-input {
            max-width: 300px;
        }

        .pagination {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .page-btn {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: color 0.15s, border-color 0.15s;
        }

        .page-btn:hover {
            color: var(--text-primary);
            border-color: var(--border-hover);
        }

        .page-btn.active {
            background-color: var(--color-purple);
            border-color: var(--color-purple);
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Task Selector layout */
        .task-grid {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
        }

        .task-list {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            max-height: 520px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem;
            background-color: var(--bg-card);
        }

        .task-item {
            height: 2.5rem;
            line-height: 2.5rem;
            padding: 0 0.8rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            color: var(--text-secondary);
            transition: background-color 0.15s, color 0.15s;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex-shrink: 0;
        }

        .task-item:hover {
            background-color: var(--bg-surface);
            color: var(--text-primary);
        }

        .task-item.active {
            background-color: var(--color-purple-glow);
            color: var(--color-purple);
            font-weight: 600;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        .task-panel-right {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
    </style>
</head>
<body>

    <!-- SIDEBAR -->
    <aside>
        <div class="sidebar-header">
            <h2>CP Calibration</h2>
            <p>Conformal Prediction Benchmark</p>
        </div>

        <!-- Metric Mode Selector -->
        <div class="filter-section">
            <span class="filter-label">Aggregation Metric</span>
            <div style="display: flex; gap: 0.5rem; background-color: var(--bg-card); padding: 0.25rem; border: 1px solid var(--border); border-radius: 8px;">
                <button id="metric-mean-btn" class="page-btn active" style="flex: 1; border: none; padding: 0.5rem; margin: 0; cursor: pointer;" onclick="setMetricMode('mean')">Mean</button>
                <button id="metric-median-btn" class="page-btn" style="flex: 1; border: none; padding: 0.5rem; margin: 0; cursor: pointer;" onclick="setMetricMode('median')">Median</button>
            </div>
        </div>

        <!-- Weight Mode Selector -->
        <div class="filter-section">
            <span class="filter-label">Aggregation Weighting</span>
            <div style="display: flex; gap: 0.5rem; background-color: var(--bg-card); padding: 0.25rem; border: 1px solid var(--border); border-radius: 8px;">
                <button id="weight-series-btn" class="page-btn active" style="flex: 1; border: none; padding: 0.5rem; margin: 0; cursor: pointer;" onclick="setWeightMode('series')">Series-Weighted</button>
                <button id="weight-task-btn" class="page-btn" style="flex: 1; border: none; padding: 0.5rem; margin: 0; cursor: pointer;" onclick="setWeightMode('task')">Task-Weighted</button>
            </div>
        </div>

        <!-- Alpha Selector -->
        <div class="filter-section">
            <span class="filter-label">Nominal Coverage (1 - α)</span>
            <select id="alpha-select">
                <!-- Dynamically Populated -->
            </select>
        </div>

        <!-- Score Type Selector -->
        <div class="filter-section">
            <span class="filter-label">Score Type</span>
            <select id="score-type-select">
                <option value="all" selected>All Score Types</option>
                <optgroup label="Physical">
                    <option value="abs">Absolute residuals</option>
                    <option value="squared">Squared residuals</option>
                    <option value="signed">Signed residuals (asymmetric)</option>
                </optgroup>
                <optgroup label="Scaled">
                    <option value="iqr_scaled">IQR-scaled residuals</option>
                    <option value="mad_scaled">MAD-scaled residuals</option>
                </optgroup>
                <optgroup label="Quantile">
                    <option value="cqr">Conformal Quantile Regression (CQR)</option>
                    <option value="scaled_cqr">Scaled CQR</option>
                    <option value="distributional">Excess mean-pinball (CRPS-like)</option>
                    <option value="cdf_tail">CDF-tail (PIT central band)</option>
                </optgroup>
                <optgroup label="Transformed / Stateful">
                    <option value="log">Log-scale residuals</option>
                    <option value="diff">Difference residuals</option>
                </optgroup>
                <optgroup label="Internal">
                    <option value="joint">Joint Methods (internal)</option>
                </optgroup>
            </select>
        </div>

        <!-- Interval Type Selector -->
        <div class="filter-section">
            <span class="filter-label">Interval Type</span>
            <select id="interval-type-select">
                <option value="all" selected>All Types</option>
                <option value="symmetric">Symmetric</option>
                <option value="asymmetric">Asymmetric</option>
            </select>
        </div>

        <!-- Step Config Selector -->
        <div class="filter-section">
            <span class="filter-label">Step Configuration</span>
            <select id="mode-select">
                <option value="all" selected>All Modes</option>
                <option value="single-step">Single-Step</option>
                <option value="multi-step">Multi-Step</option>
            </select>
        </div>

        <!-- Calibration Windows Selector -->
        <div class="filter-section">
            <span class="filter-label">Calibration Windows</span>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <select id="cal-windows-op" style="width: 70px; min-width: 70px;">
                    <option value="eq" selected>=</option>
                    <option value="lte">≤</option>
                    <option value="gte">≥</option>
                </select>
                <select id="cal-windows-select" style="flex: 1;">
                    <!-- Dynamically Populated -->
                </select>
            </div>
        </div>

        <!-- Horizon Selector -->
        <div class="filter-section">
            <span class="filter-label">Forecast Horizon</span>
            <select id="horizon-select">
                <option value="all" selected>All Horizons</option>
                <option value="short">Short</option>
                <option value="medium">Medium</option>
                <option value="long">Long</option>
            </select>
        </div>

        <!-- Model Selector -->
        <div class="filter-section">
            <span class="filter-label">Base Models</span>
            <div class="checkbox-group" id="model-checkboxes">
                <!-- Dynamically Populated -->
            </div>
        </div>

        <!-- Conformal Methods -->
        <div class="filter-section">
            <span class="filter-label">Conformal Methods</span>
            <div class="checkbox-group" id="method-checkboxes">
                <!-- Dynamically Populated -->
            </div>
        </div>
    </aside>

    <!-- MAIN CONTAINER -->
    <main>
        <!-- KPI CARDS -->
        <div class="kpi-row">
            <div class="kpi-card" style="--accent-color: var(--color-blue)">
                <span class="kpi-label">Compared Models</span>
                <span class="kpi-value" id="kpi-models">0</span>
                <span class="kpi-desc">Base forecasters evaluated</span>
            </div>
            <div class="kpi-card" style="--accent-color: var(--color-purple)">
                <span class="kpi-label">Total Time Series</span>
                <span class="kpi-value" id="kpi-configs">0</span>
                <span class="kpi-desc">Across evaluated tasks</span>
            </div>
            <div class="kpi-card" style="--accent-color: var(--color-green)">
                <span class="kpi-label">Evaluated Tasks</span>
                <span class="kpi-value" id="kpi-tasks">0</span>
                <span class="kpi-desc">Time-series datasets represented</span>
            </div>
            <div class="kpi-card" style="--accent-color: var(--color-pink)">
                <span class="kpi-label">Best Winkler Score Method</span>
                <span class="kpi-value" id="kpi-best-method" style="font-size: 1.4rem; padding: 0.3rem 0;">N/A</span>
                <span class="kpi-desc" id="kpi-best-desc">Lowest scaled Winkler score</span>
            </div>
        </div>

        <!-- TABS -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tab-global', this)">Global Aggregate Analysis</button>
            <button class="tab-btn" onclick="switchTab('tab-tasks', this)">Task-Level Drilldown</button>
            <button class="tab-btn" onclick="switchTab('tab-data', this)">Raw Data Explorer</button>
        </div>

        <!-- TAB CONTENT: GLOBAL -->
        <div class="tab-content active" id="tab-global">
            <div class="chart-grid">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Coverage vs. Average Width <span class="badge">Mean</span></span>
                    </div>
                    <div class="chart-container" id="chart-efficiency"></div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Scaled Winkler Score <span class="badge">Lower is better</span></span>
                    </div>
                    <div class="chart-container" id="chart-winkler"></div>
                </div>
            </div>
            <div class="chart-grid">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Nominal Coverage Deviation <span class="badge">Calibration Error</span></span>
                    </div>
                    <div class="chart-container" id="chart-deviation"></div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Method Scaling vs. Calibration Windows</span>
                    </div>
                    <div class="chart-container" id="chart-scaling"></div>
                </div>
            </div>
        </div>

        <!-- TAB CONTENT: TASKS -->
        <div class="tab-content" id="tab-tasks">
            <div class="task-grid">
                <div class="task-sidebar">
                    <span class="filter-label" style="display: block; margin-bottom: 0.5rem;">Select Dataset / Task</span>
                    <div class="task-list" id="task-list-container">
                        <!-- Dynamically Populated -->
                    </div>

                    <div class="card" style="margin-top: 1.5rem; padding: 1.25rem; gap: 0.75rem;">
                        <span class="filter-label" style="display: block;">Task Metadata</span>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem; color: var(--text-secondary);">
                            <div>
                                <span style="font-size: 0.75rem; color: var(--text-muted); display: block; text-transform: uppercase;">Horizon</span>
                                <strong id="meta-horizon" style="color: var(--text-primary); font-size: 1rem;">-</strong>
                            </div>
                            <div>
                                <span style="font-size: 0.75rem; color: var(--text-muted); display: block; text-transform: uppercase;">Series (N)</span>
                                <strong id="meta-series" style="color: var(--text-primary); font-size: 1rem;">-</strong>
                            </div>
                            <div>
                                <span style="font-size: 0.75rem; color: var(--text-muted); display: block; text-transform: uppercase;">Cal Windows</span>
                                <strong id="meta-cal-windows" style="color: var(--text-primary); font-size: 1rem;">-</strong>
                            </div>
                            <div>
                                <span style="font-size: 0.75rem; color: var(--text-muted); display: block; text-transform: uppercase;">Test Windows</span>
                                <strong id="meta-test-windows" style="color: var(--text-primary); font-size: 1rem;">-</strong>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="task-panel-right">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title" id="task-detail-title">Task Detail</span>
                        </div>
                        <div class="chart-grid">
                            <div class="chart-container" id="task-chart-coverage" style="height: 300px;"></div>
                            <div class="chart-container" id="task-chart-width" style="height: 300px;"></div>
                        </div>
                    </div>

                    <div class="card">
                        <span class="card-title" id="task-table-title">CP Method Performance Table</span>
                        <div class="table-wrapper">
                            <table id="task-results-table">
                               <thead>
                                   <tr>
                                       <th>Method</th>
                                       <th>Coverage</th>
                                       <th>Joint Coverage</th>
                                       <th>Avg Width</th>
                                       <th>Winkler Score</th>
                                       <th>Scaled Avg Width</th>
                                       <th>Scaled Winkler Score</th>
                                       <th>Runtime (s)</th>
                                   </tr>
                               </thead>
                               <tbody id="task-results-tbody">
                                   <!-- Dynamically Populated -->
                               </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB CONTENT: RAW DATA -->
        <div class="tab-content" id="tab-data">
            <div class="card">
                <div class="table-controls">
                    <input type="text" id="raw-search" class="search-input" placeholder="Search methods, models..." oninput="updateRawTable()">
                    <div class="pagination" id="raw-pagination">
                        <!-- Dynamically Populated -->
                    </div>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th onclick="sortRawTable('model')">Model ↕</th>
                                <th onclick="sortRawTable('score_type')">Score Type ↕</th>
                                <th onclick="sortRawTable('mode')">Mode ↕</th>
                                <th onclick="sortRawTable('horizon')">Horizon ↕</th>
                                <th onclick="sortRawTable('cal_windows')">Cal Windows ↕</th>
                                <th onclick="sortRawTable('method')">Method ↕</th>
                                <th onclick="sortRawTable('asymmetric')">Interval ↕</th>
                                <th id="th-coverage" onclick="sortRawTable('coverage_mean')">Cov Mean ↕</th>
                                <th id="th-joint" onclick="sortRawTable('joint_coverage_mean')">Joint Cov Mean ↕</th>
                                <th id="th-width" onclick="sortRawTable('scaled_avg_width_mean')">Scaled Width ↕</th>
                                <th id="th-winkler" onclick="sortRawTable('scaled_winkler_score_mean')">Scaled Winkler ↕</th>
                                <th id="th-runtime" onclick="sortRawTable('runtime_mean')">Runtime (s) ↕</th>
                            </tr>
                        </thead>
                        <tbody id="raw-tbody">
                            <!-- Dynamically Populated -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <!-- INJECT DATASET -->
    <script>
        const rawCpData = __RAW_DATA_JSON__;
    </script>

    <!-- JAVASCRIPT DASHBOARD LOGIC -->
    <script>
        // App State
        const state = {
            models: [],
            methods: [],
            tasks: [],
            metricMode: 'mean', // 'mean' or 'median'
            weightMode: 'series', // 'series' or 'task'
            filters: {
                alpha: 0.2,
                scoreType: 'all',
                calWindows: 'all',
                calWindowsOp: 'eq',
                horizon: 'all',
                mode: 'all',
                intervalType: 'all',
                models: [],
                methods: []
            },
            rawTable: {
                page: 1,
                pageSize: 15,
                search: '',
                sortBy: 'scaled_winkler_score_mean',
                sortAsc: true
            },
            selectedTask: ''
        };

        function getMetricKey(baseKey) {
            return `${baseKey}_${state.metricMode}`;
        }

        function calculateMetric(arr) {
            if (!arr || arr.length === 0) return null;
            if (state.metricMode === 'mean') {
                return arr.reduce((a, b) => a + b, 0) / arr.length;
            } else {
                const sorted = [...arr].sort((a, b) => a - b);
                const mid = Math.floor(sorted.length / 2);
                return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
            }
        }

        // Aggregate across tasks, weighting each task by its number of series
        // (n_series). Falls back to unweighted calculateMetric when no weights
        // are supplied or all weights are zero.
        function calculateWeightedMetric(values, weights) {
            if (!values || values.length === 0) return null;
            if (state.weightMode === 'task') {
                return calculateMetric(values.filter(v => v !== null && v !== undefined));
            }
            const pairs = [];
            let wSum = 0;
            for (let i = 0; i < values.length; i++) {
                const v = values[i];
                if (v === null || v === undefined) continue;
                const w = (weights && weights[i] !== null && weights[i] !== undefined) ? weights[i] : 0;
                pairs.push({ v: v, w: w });
                wSum += w;
            }
            if (pairs.length === 0) return null;
            // No usable weights -> fall back to unweighted metric.
            if (wSum <= 0) return calculateMetric(pairs.map(p => p.v));
            if (state.metricMode === 'mean') {
                let vSum = 0;
                pairs.forEach(p => { vSum += p.v * p.w; });
                return vSum / wSum;
            } else {
                // Weighted median: sort by value, find where cumulative weight
                // crosses half of the total weight.
                pairs.sort((a, b) => a.v - b.v);
                const half = wSum / 2;
                let cum = 0;
                for (let i = 0; i < pairs.length; i++) {
                    cum += pairs[i].w;
                    if (cum >= half) return pairs[i].v;
                }
                return pairs[pairs.length - 1].v;
            }
        }

        function setWeightMode(mode) {
            state.weightMode = mode;
            document.getElementById('weight-series-btn').classList.toggle('active', mode === 'series');
            document.getElementById('weight-task-btn').classList.toggle('active', mode === 'task');
            updateDashboard();
        }

        function setMetricMode(mode) {
            state.metricMode = mode;
            document.getElementById('metric-mean-btn').classList.toggle('active', mode === 'mean');
            document.getElementById('metric-median-btn').classList.toggle('active', mode === 'median');

            // Adjust raw table sort column if sorting by metric
            const metricCols = ['coverage', 'joint_coverage', 'scaled_avg_width', 'scaled_winkler_score', 'runtime'];
            metricCols.forEach(col => {
                if (state.rawTable.sortBy === `${col}_mean` && mode === 'median') {
                    state.rawTable.sortBy = `${col}_median`;
                } else if (state.rawTable.sortBy === `${col}_median` && mode === 'mean') {
                    state.rawTable.sortBy = `${col}_mean`;
                }
            });

            updateTableHeaderText();
            updateDashboard();
        }

        function updateTableHeaderText() {
            const modeCapitalized = state.metricMode === 'mean' ? 'Mean' : 'Median';
            const suffix = state.metricMode;

            const thCoverage = document.getElementById('th-coverage');
            if (thCoverage) {
                thCoverage.textContent = `Cov ${modeCapitalized} ↕`;
                thCoverage.setAttribute('onclick', `sortRawTable('coverage_${suffix}')`);
            }

            const thJoint = document.getElementById('th-joint');
            if (thJoint) {
                thJoint.textContent = `Joint Cov ${modeCapitalized} ↕`;
                thJoint.setAttribute('onclick', `sortRawTable('joint_coverage_${suffix}')`);
            }

            const thWidth = document.getElementById('th-width');
            if (thWidth) {
                thWidth.textContent = `Scaled Width ${modeCapitalized} ↕`;
                thWidth.setAttribute('onclick', `sortRawTable('scaled_avg_width_${suffix}')`);
            }

            const thWinkler = document.getElementById('th-winkler');
            if (thWinkler) {
                thWinkler.textContent = `Scaled Winkler ${modeCapitalized} ↕`;
                thWinkler.setAttribute('onclick', `sortRawTable('scaled_winkler_score_${suffix}')`);
            }

            const thRuntime = document.getElementById('th-runtime');
            if (thRuntime) {
                thRuntime.textContent = `Runtime ${modeCapitalized} (s) ↕`;
                thRuntime.setAttribute('onclick', `sortRawTable('runtime_${suffix}')`);
            }
        }

        function getAvailableCalWindows() {
            const cwSet = new Set();
            rawCpData.model_summaries.forEach(row => {
                if (row.alpha !== state.filters.alpha) return;
                if (state.filters.scoreType !== 'all' && row.score_type !== state.filters.scoreType) return;
                if (state.filters.horizon !== 'all' && row.horizon !== state.filters.horizon) return;
                if (state.filters.mode !== 'all' && row.mode !== state.filters.mode) return;
                if (state.filters.intervalType !== 'all') {
                    const wantAsymm = state.filters.intervalType === 'asymmetric';
                    if (!!row.asymmetric !== wantAsymm) return;
                }
                if (!state.filters.models.includes(row.model)) return;
                if (!state.filters.methods.includes(row.method)) return;
                if (row.cal_windows !== null && row.cal_windows !== undefined) {
                    cwSet.add(parseInt(row.cal_windows));
                }
            });
            return Array.from(cwSet).sort((a, b) => a - b);
        }

        function updateCalWindowsDropdown() {
            const available = getAvailableCalWindows();
            const selectEl = document.getElementById('cal-windows-select');
            if (!selectEl) return;

            const currentValue = state.filters.calWindows;
            let newValue = 'all';
            if (currentValue !== 'all' && available.includes(parseInt(currentValue))) {
                newValue = currentValue;
            }

            let html = `<option value="all"${newValue === 'all' ? ' selected' : ''}>All Sizes</option>`;
            available.forEach(cw => {
                html += `<option value="${cw}"${parseInt(newValue) === cw ? ' selected' : ''}>${cw} Windows</option>`;
            });

            selectEl.innerHTML = html;
            state.filters.calWindows = newValue;
        }

        const MODEL_COLORS = {
            'chronos2': '#EC4899', // pink
            'flowstate': '#3B82F6', // blue
            'timesfm': '#10B981', // green
            'tirex': '#8B5CF6'    // purple
        };

        const MODEL_SYMBOLS = {
            'chronos2': 'circle',
            'flowstate': 'square',
            'timesfm': 'diamond',
            'tirex': 'triangle-up'
        };

        const METHOD_COLORS = {
            'ACI': '#3B82F6',
            'AcMCP': '#60A5FA',
            'AgACI': '#93C5FD',
            'DtACI': '#C7D2FE',
            'Native': '#9CA3AF',
            'PID': '#10B981',
            'SplitCP': '#F59E0B',
            'TrailingWindow': '#EC4899',
            'WeightedCP': '#F43F5E',
            'JointCFRNN': '#F43F5E',
            'CopulaCPTS': '#D946EF',
            'NormMaxCP': '#A855F7',
            'CAFHT': '#8B5CF6',
            'CAFHT_PID': '#6366F1'
        };

        // Initialize App
        function initApp() {
            // Extract unique categories
            const modelSet = new Set();
            const methodSet = new Set();
            const taskSet = new Set();

            rawCpData.model_summaries.forEach(row => {
                if (row.model) modelSet.add(row.model);
                if (row.method) methodSet.add(row.method);
            });

            rawCpData.task_summaries.forEach(row => {
                if (row.task) taskSet.add(row.task);
            });

            state.models = Array.from(modelSet).sort();
            state.methods = Array.from(methodSet).sort();
            state.tasks = Array.from(taskSet).sort();

            // Set default filters
            state.filters.models = [...state.models];
            state.filters.methods = [...state.methods];

            // Render Filter controls
            renderFilterCheckboxes();

            // Dynamically filter Score Type selector to only show available ones
            const scoreTypeSet = new Set();
            rawCpData.model_summaries.forEach(row => {
                if (row.score_type) scoreTypeSet.add(row.score_type);
            });
            const scoreTypeSelectEl = document.getElementById('score-type-select');
            if (scoreTypeSelectEl) {
                const options = scoreTypeSelectEl.querySelectorAll('option');
                const existingValues = new Set();
                options.forEach(opt => {
                    if (opt.value === 'all') return;
                    existingValues.add(opt.value);
                    if (!scoreTypeSet.has(opt.value)) {
                        opt.remove();
                    }
                });

                // Dynamically add any available score types that are NOT in the pre-defined options
                let otherGroup = null;
                scoreTypeSet.forEach(st => {
                    if (!existingValues.has(st)) {
                        if (!otherGroup) {
                            otherGroup = document.createElement('optgroup');
                            otherGroup.label = "Other";
                            scoreTypeSelectEl.appendChild(otherGroup);
                        }
                        const opt = document.createElement('option');
                        opt.value = st;
                        opt.textContent = st.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                        otherGroup.appendChild(opt);
                    }
                });

                // Remove empty optgroups so we don't have empty headings
                const optgroups = scoreTypeSelectEl.querySelectorAll('optgroup');
                optgroups.forEach(group => {
                    if (group.querySelectorAll('option').length === 0) {
                        group.remove();
                    }
                });
            }

            // Dynamically populate Nominal Coverage (alpha) selector
            const alphaSet = new Set();
            rawCpData.model_summaries.forEach(row => {
                if (row.alpha !== undefined && row.alpha !== null) {
                    alphaSet.add(row.alpha);
                }
            });
            const alphas = alphaSet.size > 0 ? Array.from(alphaSet).sort((a, b) => a - b) : [0.2];
            const alphaSelectEl = document.getElementById('alpha-select');
            if (alphaSelectEl) {
                alphaSelectEl.innerHTML = alphas.map(alpha => {
                    const coverage = Math.round((1 - alpha) * 100);
                    return `<option value="${alpha}">${coverage}% (α = ${alpha})</option>`;
                }).join('');
                if (alphas.includes(0.2)) {
                    state.filters.alpha = 0.2;
                    alphaSelectEl.value = 0.2;
                } else if (alphas.length > 0) {
                    state.filters.alpha = alphas[0];
                    alphaSelectEl.value = alphas[0];
                }
            }

            // Populate task list
            populateTaskList();
            if (state.tasks.length > 0) {
                state.selectedTask = state.tasks[0];
            }

            if (alphaSelectEl) {
                alphaSelectEl.addEventListener('change', e => {
                    state.filters.alpha = parseFloat(e.target.value);
                    updateCalWindowsDropdown();
                    updateDashboard();
                });
            }

            document.getElementById('score-type-select').addEventListener('change', e => {
                state.filters.scoreType = e.target.value;
                updateCalWindowsDropdown();
                updateDashboard();
            });

            document.getElementById('interval-type-select').addEventListener('change', e => {
                state.filters.intervalType = e.target.value;
                updateCalWindowsDropdown();
                updateDashboard();
            });

            document.getElementById('cal-windows-select').addEventListener('change', e => {
                state.filters.calWindows = e.target.value;
                updateDashboard();
            });

            document.getElementById('cal-windows-op').addEventListener('change', e => {
                state.filters.calWindowsOp = e.target.value;
                updateDashboard();
            });

            document.getElementById('horizon-select').addEventListener('change', e => {
                state.filters.horizon = e.target.value;
                updateCalWindowsDropdown();
                updateDashboard();
            });

            document.getElementById('mode-select').addEventListener('change', e => {
                state.filters.mode = e.target.value;
                updateCalWindowsDropdown();
                updateDashboard();
            });

            // Run initial update
            updateCalWindowsDropdown();
            updateDashboard();
        }

        function renderFilterCheckboxes() {
            // Models
            const modelContainer = document.getElementById('model-checkboxes');
            modelContainer.innerHTML = state.models.map(m => `
                <label class="checkbox-item">
                    <input type="checkbox" value="${m}" checked onchange="toggleModelFilter('${m}', this.checked)">
                    <span>${m}</span>
                </label>
            `).join('');

            // Methods
            const methodContainer = document.getElementById('method-checkboxes');
            methodContainer.innerHTML = state.methods.map(m => `
                <label class="checkbox-item">
                    <input type="checkbox" value="${m}" checked onchange="toggleMethodFilter('${m}', this.checked)">
                    <span>${m}</span>
                </label>
            `).join('');
        }

        function populateTaskList() {
            const container = document.getElementById('task-list-container');
            container.innerHTML = state.tasks.map((task, idx) => `
                <div class="task-item ${idx === 0 ? 'active' : ''}" id="task-item-${task}" onclick="selectTask('${task}')" title="${task}">
                    ${task}
                </div>
            `).join('');
        }

        function toggleModelFilter(model, isChecked) {
            if (isChecked) {
                if (!state.filters.models.includes(model)) state.filters.models.push(model);
            } else {
                state.filters.models = state.filters.models.filter(m => m !== model);
            }
            updateCalWindowsDropdown();
            updateDashboard();
        }

        function toggleMethodFilter(method, isChecked) {
            if (isChecked) {
                if (!state.filters.methods.includes(method)) state.filters.methods.push(method);
            } else {
                state.filters.methods = state.filters.methods.filter(m => m !== method);
            }
            updateCalWindowsDropdown();
            updateDashboard();
        }

        function switchTab(tabId, btn) {
            // Toggle buttons
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');

            // Re-layout Plotly charts if switching to global or task tabs
            if (tabId === 'tab-global' || tabId === 'tab-tasks') {
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 50);
            }
        }

        function selectTask(task) {
            document.querySelectorAll('.task-item').forEach(item => item.classList.remove('active'));
            document.getElementById(`task-item-${task}`).classList.add('active');
            state.selectedTask = task;
            updateTaskTab();
        }

        // Get Filtered Model Summaries
        function getFilteredModelSummaries() {
            return rawCpData.model_summaries.filter(row => {
                if (row.alpha !== state.filters.alpha) return false;
                if (state.filters.scoreType !== 'all' && row.score_type !== state.filters.scoreType) return false;
                if (state.filters.calWindows !== 'all') {
                    const selVal = parseInt(state.filters.calWindows);
                    const rowVal = parseInt(row.cal_windows);
                    if (state.filters.calWindowsOp === 'eq' && rowVal !== selVal) return false;
                    if (state.filters.calWindowsOp === 'lte' && rowVal > selVal) return false;
                    if (state.filters.calWindowsOp === 'gte' && rowVal < selVal) return false;
                }
                if (state.filters.horizon !== 'all' && row.horizon !== state.filters.horizon) return false;
                if (state.filters.mode !== 'all' && row.mode !== state.filters.mode) return false;
                if (state.filters.intervalType !== 'all') {
                    const wantAsymm = state.filters.intervalType === 'asymmetric';
                    if (!!row.asymmetric !== wantAsymm) return false;
                }
                if (!state.filters.models.includes(row.model)) return false;
                if (!state.filters.methods.includes(row.method)) return false;
                return true;
            });
        }

        // Update KPI card metrics
        function updateKPIs(filteredSummaries) {
            // Models Count
            const models = new Set(filteredSummaries.map(r => r.model));
            document.getElementById('kpi-models').textContent = models.size;

            // Tasks represented and Total Time Series Count
            const taskSeriesMap = new Map();
            rawCpData.task_summaries.forEach(r => {
                if (r.alpha !== state.filters.alpha) return;
                if (state.filters.scoreType !== 'all' && r.score_type !== state.filters.scoreType) return;
                if (state.filters.calWindows !== 'all') {
                    const selVal = parseInt(state.filters.calWindows);
                    const rowVal = parseInt(r.cal_windows);
                    if (state.filters.calWindowsOp === 'eq' && rowVal !== selVal) return;
                    if (state.filters.calWindowsOp === 'lte' && rowVal > selVal) return;
                    if (state.filters.calWindowsOp === 'gte' && rowVal < selVal) return;
                }
                if (state.filters.horizon !== 'all' && r.horizon !== state.filters.horizon) return;
                if (state.filters.mode !== 'all' && r.mode !== state.filters.mode) return;
                if (state.filters.intervalType !== 'all') {
                    const wantAsymm = state.filters.intervalType === 'asymmetric';
                    if (!!r.asymmetric !== wantAsymm) return;
                }
                if (!state.filters.models.includes(r.model)) return;
                if (!state.filters.methods.includes(r.method)) return;
                taskSeriesMap.set(r.task, r.n_series || 0);
            });

            document.getElementById('kpi-tasks').textContent = taskSeriesMap.size;

            let totalSeries = 0;
            taskSeriesMap.forEach(n => {
                totalSeries += n;
            });
            document.getElementById('kpi-configs').textContent = totalSeries;

            // Best Winkler Score method (lowest scaled winkler score).
            // Aggregate across tasks weighted by the number of series per task
            // so larger tasks contribute proportionally more than small ones.
            const methodScores = {};
            const methodWeights = {};
            const winklerKey = getMetricKey('scaled_winkler_score');
            filteredSummaries.forEach(r => {
                if (r[winklerKey] !== null) {
                    if (!methodScores[r.method]) {
                        methodScores[r.method] = [];
                        methodWeights[r.method] = [];
                    }
                    methodScores[r.method].push(r[winklerKey]);
                    methodWeights[r.method].push(r.n_series || 0);
                }
            });

            let bestMethod = 'N/A';
            let lowestScore = Infinity;
            Object.keys(methodScores).forEach(m => {
                const val = calculateWeightedMetric(methodScores[m], methodWeights[m]);
                if (val !== null && val < lowestScore) {
                    lowestScore = val;
                    bestMethod = m;
                }
            });

            document.getElementById('kpi-best-method').textContent = bestMethod;
            const weightLabel = state.weightMode === 'series' ? 'series-weighted' : 'task-weighted';
            document.getElementById('kpi-best-desc').textContent = `Lowest ${weightLabel} ${state.metricMode} scaled Winkler score`;
        }

        // Main Draw Loop
        function updateDashboard() {
            const filteredData = getFilteredModelSummaries();
            updateKPIs(filteredData);

            // Draw Global Charts
            drawEfficiencyScatter(filteredData);
            drawWinklerBar(filteredData);
            drawDeviationBar(filteredData);
            drawScalingLine(filteredData);

            // Draw Task tab
            updateTaskTab();

            // Update Explorer raw table
            updateRawTable();
        }

        // Chart 1: Coverage vs Scaled Average Width Scatter Plot
        function drawEfficiencyScatter(data) {
            // Group by model + method to display aggregates
            const groups = {};
            const covKey = getMetricKey('coverage');
            const widthKey = getMetricKey('scaled_avg_width');
            const winklerKey = getMetricKey('scaled_winkler_score');

            data.forEach(r => {
                const key = `${r.model} | ${r.method}`;
                if (!groups[key]) {
                    groups[key] = {
                        model: r.model,
                        method: r.method,
                        coverage: [],
                        coverageWeights: [],
                        width: [],
                        widthWeights: [],
                        winkler: [],
                        winklerWeights: []
                    };
                }
                const weight = r.n_series || 0;
                if (r[covKey] !== null) {
                    groups[key].coverage.push(r[covKey]);
                    groups[key].coverageWeights.push(weight);
                }
                if (r[widthKey] !== null) {
                    groups[key].width.push(r[widthKey]);
                    groups[key].widthWeights.push(weight);
                }
                if (r[winklerKey] !== null) {
                    groups[key].winkler.push(r[winklerKey]);
                    groups[key].winklerWeights.push(weight);
                }
            });

            // Map methods to traces for nice color grouping by method and symbol by model
            const traces = state.filters.methods.map(method => {
                const x = [];
                const y = [];
                const text = [];
                const symbols = [];

                Object.values(groups).forEach(g => {
                    if (g.method === method) {
                        if (g.coverage.length === 0 || g.width.length === 0) return;
                        const metricCoverage = calculateWeightedMetric(g.coverage, g.coverageWeights);
                        const metricWidth = calculateWeightedMetric(g.width, g.widthWeights);
                        const winklerVal = calculateWeightedMetric(g.winkler, g.winklerWeights);
                        const winklerText = winklerVal !== null ? winklerVal.toFixed(3) : 'N/A';

                        x.push(metricCoverage);
                        y.push(metricWidth);
                        text.push(`${g.model} | ${g.method} (Cov: ${metricCoverage.toFixed(3)}, Width: ${metricWidth.toFixed(3)}, Winkler: ${winklerText})`);
                        symbols.push(MODEL_SYMBOLS[g.model] || 'circle');
                    }
                });

                if (x.length === 0) return null;

                return {
                    x: x,
                    y: y,
                    text: text,
                    mode: 'markers',
                    name: method,
                    marker: {
                        color: METHOD_COLORS[method] || '#FFF',
                        symbol: symbols,
                        size: 12,
                        line: { color: '#000', width: 1 }
                    },
                    type: 'scatter'
                };
            }).filter(t => t !== null);

            // Add shapes for Nominal Coverage target line
            const weightLabel = state.weightMode === 'series' ? 'Series-Weighted' : 'Task-Weighted';
            const modeLabel = state.metricMode === 'mean' ? 'Mean' : 'Median';
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0E131F',
                xaxis: {
                    title: `${weightLabel} ${modeLabel} Coverage`,
                    gridcolor: '#222D44',
                    zeroline: false,
                    tickformat: '.2f',
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                yaxis: {
                    title: `Scaled ${weightLabel} ${modeLabel} Avg Width`,
                    gridcolor: '#222D44',
                    zeroline: false,
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                margin: { l: 50, r: 20, t: 30, b: 50 },
                legend: { font: { color: '#F3F4F6' } },
                shapes: [
                    {
                        type: 'line',
                        x0: 1 - state.filters.alpha,
                        y0: 0,
                        x1: 1 - state.filters.alpha,
                        y1: 1,
                        yref: 'paper',
                        line: {
                            color: '#EF4444',
                            width: 1.5,
                            dash: 'dash'
                        }
                    }
                ],
                annotations: [
                    {
                        x: 1 - state.filters.alpha,
                        y: 0.95,
                        yref: 'paper',
                        text: `Target Nominal Coverage (${Math.round((1 - state.filters.alpha) * 100)}%)`,
                        showarrow: false,
                        textangle: -90,
                        xanchor: 'right',
                        font: { color: '#EF4444', size: 10 }
                    }
                ]
            };

            Plotly.newPlot('chart-efficiency', traces, layout, { responsive: true, displayModeBar: false });
        }

        // Chart 2: Mean/Median Scaled Winkler Score Bar Chart
        function drawWinklerBar(data) {
            // Group by method and model, keeping per-task series counts as weights
            const groups = {};
            const winklerKey = getMetricKey('scaled_winkler_score');
            data.forEach(r => {
                if (r[winklerKey] === null) return;
                const key = `${r.model} | ${r.method}`;
                if (!groups[key]) {
                    groups[key] = { values: [], weights: [] };
                }
                groups[key].values.push(r[winklerKey]);
                groups[key].weights.push(r.n_series || 0);
            });

            // Construct trace for each CP method (X-axis is base models).
            // Per-task scores are aggregated weighted by n_series so each task
            // contributes in proportion to its number of series.
            const traces = state.filters.methods.map(method => {
                const xValues = state.filters.models;
                const yValues = xValues.map(model => {
                    const key = `${model} | ${method}`;
                    const g = groups[key];
                    if (!g || g.values.length === 0) return null;
                    return calculateWeightedMetric(g.values, g.weights);
                });

                return {
                    x: xValues,
                    y: yValues,
                    name: method,
                    type: 'bar',
                    marker: { color: METHOD_COLORS[method] || '#FFF' }
                };
            }).filter(t => t.y.some(v => v !== null));

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0E131F',
                barmode: 'group',
                xaxis: {
                    title: 'Base Model',
                    gridcolor: '#222D44',
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                yaxis: {
                    title: `${state.weightMode === 'series' ? 'Series-Weighted' : 'Task-Weighted'} ${state.metricMode === 'mean' ? 'Mean' : 'Median'} Scaled Winkler Score`,
                    gridcolor: '#222D44',
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                margin: { l: 50, r: 20, t: 30, b: 50 },
                legend: { font: { color: '#F3F4F6' } }
            };

            Plotly.newPlot('chart-winkler', traces, layout, { responsive: true, displayModeBar: false });
        }

        // Chart 3: Calibration Error (Coverage Deviation) Bar Chart
        function drawDeviationBar(data) {
            // Group by method
            const methodDevs = {};
            const methodWeights = {};
            const covKey = getMetricKey('coverage');
            data.forEach(r => {
                if (r[covKey] === null) return;
                const dev = Math.abs(r[covKey] - (1 - state.filters.alpha));
                if (!methodDevs[r.method]) {
                    methodDevs[r.method] = [];
                    methodWeights[r.method] = [];
                }
                methodDevs[r.method].push(dev);
                methodWeights[r.method].push(r.n_series || 0);
            });

            const xValues = state.filters.methods.filter(m => methodDevs[m] && methodDevs[m].length > 0);
            const yValues = xValues.map(m => {
                return calculateWeightedMetric(methodDevs[m], methodWeights[m]);
            });

            const trace = {
                x: xValues,
                y: yValues,
                type: 'bar',
                marker: {
                    color: xValues.map(m => METHOD_COLORS[m] || '#8B5CF6'),
                    line: { color: '#222D44', width: 1 }
                }
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0E131F',
                xaxis: {
                    title: 'Conformal Method',
                    gridcolor: '#222D44',
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                yaxis: {
                    title: `${state.weightMode === 'series' ? 'Series-Weighted' : 'Task-Weighted'} ${state.metricMode === 'mean' ? 'Mean' : 'Median'} Absolute Deviation from ${Math.round((1 - state.filters.alpha) * 100)}%`,
                    gridcolor: '#222D44',
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                margin: { l: 50, r: 20, t: 30, b: 50 }
            };

            Plotly.newPlot('chart-deviation', [trace], layout, { responsive: true, displayModeBar: false });
        }

        // Chart 4: Calibration Window Scaling Behavior Line Chart
        function drawScalingLine(data) {
            // Group by cal_windows and method
            const groups = {};
            const covKey = getMetricKey('coverage');
            data.forEach(r => {
                if (r.cal_windows === null || r[covKey] === null) return;
                const cw = parseInt(r.cal_windows);
                if (!groups[r.method]) groups[r.method] = {};
                if (!groups[r.method][cw]) groups[r.method][cw] = { values: [], weights: [] };
                groups[r.method][cw].values.push(r[covKey]);
                groups[r.method][cw].weights.push(r.n_series || 0);
            });

            // Get sorted list of calibration window values
            const allCws = [2, 5, 6, 8, 10, 15, 20, 50, 100];

            const traces = state.filters.methods.map(method => {
                const methodGroup = groups[method];
                if (!methodGroup) return null;

                const xValues = [];
                const yValues = [];

                allCws.forEach(cw => {
                    if (methodGroup[cw] && methodGroup[cw].values.length > 0) {
                        xValues.push(cw);
                        yValues.push(calculateWeightedMetric(methodGroup[cw].values, methodGroup[cw].weights));
                    }
                });

                if (xValues.length === 0) return null;

                return {
                    x: xValues,
                    y: yValues,
                    name: method,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { width: 2, color: METHOD_COLORS[method] },
                    marker: { size: 6 }
                };
            }).filter(t => t !== null);

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0E131F',
                xaxis: {
                    title: 'Calibration Windows Size (cal_windows)',
                    type: 'log',
                    gridcolor: '#222D44',
                    tickvals: allCws,
                    ticktext: allCws.map(String),
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' }
                },
                yaxis: {
                    title: `${state.weightMode === 'series' ? 'Series-Weighted' : 'Task-Weighted'} ${state.metricMode === 'mean' ? 'Mean' : 'Median'} Coverage`,
                    gridcolor: '#222D44',
                    tickfont: { color: '#9CA3AF' },
                    titlefont: { color: '#F3F4F6' },
                    range: [0.3, 1.05]
                },
                margin: { l: 50, r: 20, t: 30, b: 50 },
                legend: { font: { color: '#F3F4F6' } },
                shapes: [
                    {
                        type: 'line',
                        x0: 2,
                        y0: 1 - state.filters.alpha,
                        x1: 100,
                        y1: 1 - state.filters.alpha,
                        line: {
                            color: '#EF4444',
                            width: 1.5,
                            dash: 'dash'
                        }
                    }
                ]
            };

            Plotly.newPlot('chart-scaling', traces, layout, { responsive: true, displayModeBar: false });
        }

        // UPDATE TASK TAB
        function updateTaskTab() {
            const task = state.selectedTask;
            if (!task) return;

            document.getElementById('task-detail-title').innerHTML = `Task Analysis: <strong>${task}</strong>`;

            // Populate metadata
            if (rawCpData.tasks_metadata && rawCpData.tasks_metadata[task]) {
                const meta = rawCpData.tasks_metadata[task];
                document.getElementById('meta-horizon').textContent = meta.horizon !== null ? meta.horizon : '-';
                document.getElementById('meta-series').textContent = meta.n_series !== null ? meta.n_series : '-';
                document.getElementById('meta-cal-windows').textContent = meta.cal_windows !== null ? meta.cal_windows : '-';
                document.getElementById('meta-test-windows').textContent = meta.test_windows !== null ? meta.test_windows : '-';
            } else {
                document.getElementById('meta-horizon').textContent = '-';
                document.getElementById('meta-series').textContent = '-';
                document.getElementById('meta-cal-windows').textContent = '-';
                document.getElementById('meta-test-windows').textContent = '-';
            }

            // Set dynamic table title based on metricMode
            document.getElementById('task-table-title').textContent = `CP Method Performance Table (${state.metricMode === 'mean' ? 'Mean' : 'Median'})`;

            // Filter task summaries
            const taskRows = rawCpData.task_summaries.filter(r => {
                if (r.task !== task) return false;
                if (r.alpha !== state.filters.alpha) return false;
                if (!state.filters.models.includes(r.model)) return false;
                if (!state.filters.methods.includes(r.method)) return false;
                if (state.filters.scoreType !== 'all' && r.score_type !== state.filters.scoreType) return false;
                if (state.filters.calWindows !== 'all') {
                    const selVal = parseInt(state.filters.calWindows);
                    const rowVal = parseInt(r.cal_windows);
                    if (state.filters.calWindowsOp === 'eq' && rowVal !== selVal) return false;
                    if (state.filters.calWindowsOp === 'lte' && rowVal > selVal) return false;
                    if (state.filters.calWindowsOp === 'gte' && rowVal < selVal) return false;
                }
                if (state.filters.horizon !== 'all' && r.horizon !== state.filters.horizon) return false;
                if (state.filters.mode !== 'all' && r.mode !== state.filters.mode) return false;
                if (state.filters.intervalType !== 'all') {
                    const wantAsymm = state.filters.intervalType === 'asymmetric';
                    if (!!r.asymmetric !== wantAsymm) return false;
                }
                return true;
            });

            // Group by method to aggregate over models
            const methodsData = {};
            taskRows.forEach(r => {
                if (!methodsData[r.method]) {
                    methodsData[r.method] = {
                        method: r.method,
                        coverage: [],
                        avg_width: [],
                        winkler_score: [],
                        scaled_avg_width: [],
                        scaled_winkler_score: [],
                        runtime: [],
                        joint_coverage: []
                    };
                }
                if (r.coverage !== null) methodsData[r.method].coverage.push(r.coverage);
                if (r.joint_coverage !== null) methodsData[r.method].joint_coverage.push(r.joint_coverage);
                if (r.avg_width !== null) methodsData[r.method].avg_width.push(r.avg_width);
                if (r.winkler_score !== null) methodsData[r.method].winkler_score.push(r.winkler_score);
                if (r.scaled_avg_width !== null) methodsData[r.method].scaled_avg_width.push(r.scaled_avg_width);
                if (r.scaled_winkler_score !== null) methodsData[r.method].scaled_winkler_score.push(r.scaled_winkler_score);
                if (r.runtime !== null) methodsData[r.method].runtime.push(r.runtime);
            });

            const tableRows = [];
            const sortedMethods = Object.keys(methodsData).sort();

            const chartMethods = [];
            const chartCoverages = [];
            const chartWinklerScores = [];

            sortedMethods.forEach(method => {
                const mData = methodsData[method];
                const metricCoverage = calculateMetric(mData.coverage);
                const metricJoint = calculateMetric(mData.joint_coverage);
                const metricWidth = calculateMetric(mData.avg_width);
                const metricWinkler = calculateMetric(mData.winkler_score);
                const metricScaledWidth = calculateMetric(mData.scaled_avg_width);
                const metricScaledWinkler = calculateMetric(mData.scaled_winkler_score);
                const metricRuntime = mData.runtime.length > 0 ? mData.runtime.reduce((a, b) => a + b, 0) : null;

                chartMethods.push(method);
                chartCoverages.push(metricCoverage);
                chartWinklerScores.push(metricScaledWinkler);

                // Add to table
                const covClass = (metricCoverage !== null && metricCoverage >= 0.78) ? 'coverage-good' : 'coverage-bad';

                tableRows.push(`
                    <tr>
                        <td style="font-weight: 600; color: #FFF; font-family: var(--font-sans);">${method}</td>
                        <td class="${covClass}">${metricCoverage !== null ? metricCoverage.toFixed(4) : '-'}</td>
                        <td>${metricJoint !== null ? metricJoint.toFixed(4) : '-'}</td>
                        <td>${metricWidth !== null ? metricWidth.toFixed(2) : '-'}</td>
                        <td>${metricWinkler !== null ? metricWinkler.toFixed(2) : '-'}</td>
                        <td>${metricScaledWidth !== null ? metricScaledWidth.toFixed(4) : '-'}</td>
                        <td style="color: var(--color-purple); font-weight: 600;">${metricScaledWinkler !== null ? metricScaledWinkler.toFixed(4) : '-'}</td>
                        <td>${metricRuntime !== null ? metricRuntime.toFixed(5) : '-'}</td>
                    </tr>
                `);
            });

            document.getElementById('task-results-tbody').innerHTML = tableRows.join('');

            // Draw Coverage Bar chart for Task
            const coverageTrace = {
                x: chartMethods,
                y: chartCoverages,
                type: 'bar',
                name: 'Coverage',
                marker: { color: chartMethods.map(m => METHOD_COLORS[m] || '#8B5CF6') }
            };

            const coverageLayout = {
                title: { text: `Coverage per Method (${state.metricMode === 'mean' ? 'Mean' : 'Median'})`, font: { color: '#FFF', size: 13 } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0E131F',
                xaxis: { tickfont: { color: '#9CA3AF' } },
                yaxis: { title: 'Coverage', gridcolor: '#222D44', range: [0, 1.05], tickfont: { color: '#9CA3AF' } },
                margin: { l: 40, r: 20, t: 40, b: 30 },
                shapes: [
                    {
                        type: 'line',
                        x0: -0.5,
                        y0: 1 - state.filters.alpha,
                        x1: chartMethods.length - 0.5,
                        y1: 1 - state.filters.alpha,
                        line: {
                            color: '#EF4444',
                            width: 1.5,
                            dash: 'dash'
                        }
                    }
                ]
            };

            Plotly.newPlot('task-chart-coverage', [coverageTrace], coverageLayout, { responsive: true, displayModeBar: false });

            // Draw Winkler Bar chart for Task
            const winklerTrace = {
                x: chartMethods,
                y: chartWinklerScores,
                type: 'bar',
                name: 'Scaled Winkler Score',
                marker: { color: chartMethods.map(m => METHOD_COLORS[m] || '#8B5CF6') }
            };

            const winklerLayout = {
                title: { text: `Scaled Winkler Score (${state.metricMode === 'mean' ? 'Mean' : 'Median'}, Lower is better)`, font: { color: '#FFF', size: 13 } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0E131F',
                xaxis: { tickfont: { color: '#9CA3AF' } },
                yaxis: { title: 'Scaled Winkler Score', gridcolor: '#222D44', tickfont: { color: '#9CA3AF' } },
                margin: { l: 40, r: 20, t: 40, b: 30 }
            };

            Plotly.newPlot('task-chart-width', [winklerTrace], winklerLayout, { responsive: true, displayModeBar: false });
        }

        // RAW DATA EXPLORER TAB LOGIC
        let rawTableFilteredData = [];

        function sortRawTable(col) {
            if (state.rawTable.sortBy === col) {
                state.rawTable.sortAsc = !state.rawTable.sortAsc;
            } else {
                state.rawTable.sortBy = col;
                state.rawTable.sortAsc = true;
            }
            updateRawTable();
        }

        function updateRawTable() {
            const search = document.getElementById('raw-search').value.toLowerCase();
            const summaries = getFilteredModelSummaries();

            // Filter by search
            rawTableFilteredData = summaries.filter(row => {
                if (search === '') return true;
                const matchesModel = row.model.toLowerCase().includes(search);
                const matchesMethod = row.method.toLowerCase().includes(search);
                const matchesScore = row.score_type.toLowerCase().includes(search);
                const matchesMode = (row.mode || '').toLowerCase().includes(search);
                return matchesModel || matchesMethod || matchesScore || matchesMode;
            });

            // Sort
            rawTableFilteredData.sort((a, b) => {
                let valA = a[state.rawTable.sortBy];
                let valB = b[state.rawTable.sortBy];

                if (valA === null || valA === undefined) return 1;
                if (valB === null || valB === undefined) return -1;

                if (typeof valA === 'string') {
                    return state.rawTable.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
                } else {
                    return state.rawTable.sortAsc ? valA - valB : valB - valA;
                }
            });

            // Render Page
            renderRawTablePage();
        }

        function renderRawTablePage() {
            const page = state.rawTable.page;
            const pageSize = state.rawTable.pageSize;
            const startIndex = (page - 1) * pageSize;
            const endIndex = Math.min(startIndex + pageSize, rawTableFilteredData.length);

            const pagedData = rawTableFilteredData.slice(startIndex, endIndex);

            const tbody = document.getElementById('raw-tbody');
            if (pagedData.length === 0) {
                tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; color: var(--text-muted);">No records match your filters.</td></tr>`;
                document.getElementById('raw-pagination').innerHTML = '';
                return;
            }

            const covKey = getMetricKey('coverage');
            const jointKey = getMetricKey('joint_coverage');
            const widthKey = getMetricKey('scaled_avg_width');
            const winklerKey = getMetricKey('scaled_winkler_score');
            const runtimeKey = getMetricKey('runtime');

            tbody.innerHTML = pagedData.map(row => {
                const covVal = row[covKey];
                const jointVal = row[jointKey];
                const widthVal = row[widthKey];
                const winklerVal = row[winklerKey];
                const runtimeVal = row[runtimeKey];
                const covClass = (covVal !== null && covVal >= 0.78) ? 'coverage-good' : 'coverage-bad';

                return `
                    <tr>
                        <td>${row.model}</td>
                        <td>${row.score_type}</td>
                        <td>${row.mode || '-'}</td>
                        <td>${row.horizon || '-'}</td>
                        <td>${row.cal_windows}</td>
                        <td style="color: #FFF; font-weight: 500;">${row.method}</td>
                        <td>${row.asymmetric ? 'Asymmetric' : 'Symmetric'}</td>
                        <td class="${covClass}">${covVal !== null ? covVal.toFixed(4) : '-'}</td>
                        <td>${jointVal !== null ? jointVal.toFixed(4) : '-'}</td>
                        <td>${widthVal !== null ? widthVal.toFixed(4) : '-'}</td>
                        <td style="color: var(--color-purple); font-weight: 600;">${winklerVal !== null ? winklerVal.toFixed(4) : '-'}</td>
                        <td>${runtimeVal !== null ? runtimeVal.toFixed(5) : '-'}</td>
                    </tr>
                `;
            }).join('');

            // Pagination Controls
            const totalPages = Math.ceil(rawTableFilteredData.length / pageSize);
            const pagContainer = document.getElementById('raw-pagination');

            let pagHtml = [];
            // Prev
            pagHtml.push(`<button class="page-btn" ${page === 1 ? 'disabled' : ''} onclick="changeRawPage(${page - 1})">Prev</button>`);

            // Current / Total info
            pagHtml.push(`<span style="font-size: 0.8rem; color: var(--text-secondary); margin: 0 0.5rem;">Page ${page} of ${totalPages || 1} (${rawTableFilteredData.length} entries)</span>`);

            // Next
            pagHtml.push(`<button class="page-btn" ${page === totalPages || totalPages === 0 ? 'disabled' : ''} onclick="changeRawPage(${page + 1})">Next</button>`);

            pagContainer.innerHTML = pagHtml.join('');
        }

        function changeRawPage(p) {
            state.rawTable.page = p;
            renderRawTablePage();
        }

        // Bootstrap the app
        window.addEventListener('DOMContentLoaded', initApp);
    </script>
</body>
</html>
""".replace('__RAW_DATA_JSON__', json.dumps(data))

    output_path = pathlib.Path('results/cp_dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(
        f'Interactive CP Dashboard compiled successfully and written to {output_path.absolute()}'
    )


if __name__ == '__main__':
    main()
