import os
import subprocess

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Forest Fire Early-Warning System - Comprehensive Guide</title>
<style>
    @page {
        size: A4;
        margin: 16mm 14mm 16mm 14mm;
        @bottom-right {
            content: "Page " counter(page);
            font-size: 9pt;
            color: #718096;
        }
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #2D3748;
        line-height: 1.55;
        font-size: 10pt;
        background: #FFFFFF;
        margin: 0;
        padding: 0;
    }

    h1, h2, h3, h4 {
        color: #1A202C;
        font-weight: 700;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
        page-break-after: avoid;
    }

    h1 {
        font-size: 22pt;
        color: #C53030;
        border-bottom: 3px solid #E53E3E;
        padding-bottom: 6px;
        margin-top: 0;
    }

    .subtitle {
        font-size: 12pt;
        color: #4A5568;
        font-weight: 500;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    h2 {
        font-size: 14pt;
        color: #2B6CB0;
        border-bottom: 1.5px solid #E2E8F0;
        padding-bottom: 4px;
        margin-top: 1.4em;
    }

    h3 {
        font-size: 11pt;
        color: #2D3748;
        margin-top: 1em;
    }

    p {
        margin: 0.5em 0;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 8.5pt;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-red { background: #FED7D7; color: #9B2C2C; }
    .badge-yellow { background: #FEFCBF; color: #975A16; }
    .badge-green { background: #C6F6D5; color: #22543D; }
    .badge-blue { background: #BEE3F8; color: #2B6CB0; }

    .callout {
        background: #F7FAFC;
        border-left: 4px solid #3182CE;
        padding: 10px 14px;
        margin: 12px 0;
        border-radius: 0 6px 6px 0;
        font-size: 9.5pt;
    }

    .callout-warning {
        background: #FFFAF0;
        border-left-color: #DD6B20;
    }

    .grid-2 {
        display: flex;
        gap: 12px;
        margin: 10px 0;
    }

    .card {
        flex: 1;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 12px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        page-break-inside: avoid;
    }

    .card-header {
        font-weight: bold;
        font-size: 10.5pt;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }

    th, td {
        border: 1px solid #CBD5E0;
        padding: 6px 10px;
        text-align: left;
    }

    th {
        background: #EDF2F7;
        font-weight: 600;
        color: #2D3748;
    }

    tr:nth-child(even) td {
        background: #F7FAFC;
    }

    .diagram-container {
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
    }

    svg {
        max-width: 100%;
        height: auto;
    }

    .page-break {
        page-break-before: always;
    }

    .code-span {
        font-family: Consolas, "Courier New", monospace;
        background: #EDF2F7;
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 8.5pt;
        color: #805AD5;
    }
</style>
</head>
<body>

<!-- PAGE 1: TITLE & CORE CONCEPT -->
<h1>Multi-Modal Forest Fire Early-Warning System</h1>
<div class="subtitle">Complete Technical Guide & System Architecture &bull; Western Maharashtra (Sahyadri Belt)</div>

<div class="callout">
    <strong>Executive Summary:</strong> Wildfires spread exponentially, leaving narrow margins for human intervention. Single-sensor monitoring systems suffer from high false alarm rates or dangerous detection delays. This project implements a fault-tolerant, 4-layer AI multi-modal early-warning network that fuses weather patterns, orbital satellite thermal hotspots, ground-level IoT sensors, and computer vision classification into a unified confidence metric.
</div>

<h2>1. The Problem & Intuitive Solution</h2>
<p>
Forest environments are unpredictable. If a system relies on only one type of sensor, failure is inevitable:
</p>
<ul>
    <li><strong>Thermometers alone fail:</strong> A normal 42&deg;C summer afternoon can trigger false heat alarms without fire.</li>
    <li><strong>Cameras alone fail:</strong> Morning mountain mist, cloud shadows, or dust storms are often falsely identified as smoke columns.</li>
    <li><strong>Satellites alone fail:</strong> Earth-observation satellites pass over a specific region only 2 to 4 times per day. A fire starting between passes can burn unnoticed for hours.</li>
    <li><strong>IoT probes alone fail:</strong> IoT sensors only monitor a radius of 20&ndash;50 meters. Complete coverage of vast wilderness is financially impossible.</li>
</ul>

<div class="callout callout-warning">
    <strong>The Solution (The "Council of Detectives"):</strong> The system combines four completely independent modalities into a single Fusion Engine. If one layer goes down (e.g., IoT battery death or delayed satellite imagery), the remaining layers automatically redistribute weights proportionally.
</div>

<h2>2. End-to-End System Architecture</h2>
<div class="diagram-container">
    <svg viewBox="0 0 760 380" xmlns="http://www.w3.org/2000/svg" style="font-family: sans-serif;">
        <!-- Inputs Column -->
        <rect x="10" y="20" width="150" height="60" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1.5"/>
        <text x="85" y="45" font-size="11" font-weight="bold" text-anchor="middle" fill="#2D3748">Weather Stations</text>
        <text x="85" y="62" font-size="9" fill="#718096" text-anchor="middle">Temp, RH, Wind, Rain</text>

        <rect x="10" y="110" width="150" height="60" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1.5"/>
        <text x="85" y="135" font-size="11" font-weight="bold" text-anchor="middle" fill="#2D3748">NASA Satellites</text>
        <text x="85" y="152" font-size="9" fill="#718096" text-anchor="middle">VIIRS 375m &amp; MODIS</text>

        <rect x="10" y="200" width="150" height="60" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1.5"/>
        <text x="85" y="225" font-size="11" font-weight="bold" text-anchor="middle" fill="#2D3748">Field Cameras</text>
        <text x="85" y="242" font-size="9" fill="#718096" text-anchor="middle">RGB Watchtower Video</text>

        <rect x="10" y="290" width="150" height="60" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1.5"/>
        <text x="85" y="315" font-size="11" font-weight="bold" text-anchor="middle" fill="#2D3748">IoT Ground Probes</text>
        <text x="85" y="332" font-size="9" fill="#718096" text-anchor="middle">Smoke PPM, Temp, RH</text>

        <!-- Arrows to Layers -->
        <path d="M 160 50 L 210 50" stroke="#718096" stroke-width="2" marker-end="url(#arrow)"/>
        <path d="M 160 140 L 210 140" stroke="#718096" stroke-width="2" marker-end="url(#arrow)"/>
        <path d="M 160 230 L 210 230" stroke="#718096" stroke-width="2" marker-end="url(#arrow)"/>
        <path d="M 160 320 L 210 320" stroke="#718096" stroke-width="2" marker-end="url(#arrow)"/>

        <!-- Layer Blocks -->
        <rect x="210" y="20" width="170" height="60" rx="6" fill="#EBF8FF" stroke="#3182CE" stroke-width="1.5"/>
        <text x="295" y="43" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#2B6CB0">Layer 1: Weather Risk</text>
        <text x="295" y="58" font-size="8.5" fill="#4A5568" text-anchor="middle">RandomForest Classifier</text>
        <text x="295" y="71" font-size="8" font-weight="bold" fill="#3182CE" text-anchor="middle">Weight: 15%</text>

        <rect x="210" y="110" width="170" height="60" rx="6" fill="#EBF8FF" stroke="#3182CE" stroke-width="1.5"/>
        <text x="295" y="133" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#2B6CB0">Layer 2: Satellite FIRMS</text>
        <text x="295" y="148" font-size="8.5" fill="#4A5568" text-anchor="middle">Thermal Hotspot Ingestion</text>
        <text x="295" y="161" font-size="8" font-weight="bold" fill="#3182CE" text-anchor="middle">Weight: 25%</text>

        <rect x="210" y="200" width="170" height="60" rx="6" fill="#EBF8FF" stroke="#3182CE" stroke-width="1.5"/>
        <text x="295" y="223" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#2B6CB0">Layer 3: Vision CNN</text>
        <text x="295" y="238" font-size="8.5" fill="#4A5568" text-anchor="middle">EfficientNetB3 Deep Net</text>
        <text x="295" y="251" font-size="8" font-weight="bold" fill="#3182CE" text-anchor="middle">Weight: 30%</text>

        <rect x="210" y="290" width="170" height="60" rx="6" fill="#EBF8FF" stroke="#3182CE" stroke-width="1.5"/>
        <text x="295" y="313" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#2B6CB0">Layer 4: IoT Pipeline</text>
        <text x="295" y="328" font-size="8.5" fill="#4A5568" text-anchor="middle">Isolation Forest Anomaly</text>
        <text x="295" y="341" font-size="8" font-weight="bold" fill="#3182CE" text-anchor="middle">Weight: 30%</text>

        <!-- Convergence to Fusion Engine -->
        <path d="M 380 50 L 440 160" stroke="#3182CE" stroke-width="1.5"/>
        <path d="M 380 140 L 440 175" stroke="#3182CE" stroke-width="1.5"/>
        <path d="M 380 230 L 440 195" stroke="#3182CE" stroke-width="1.5"/>
        <path d="M 380 320 L 440 210" stroke="#3182CE" stroke-width="1.5"/>

        <!-- Fusion Engine Box -->
        <rect x="440" y="140" width="150" height="90" rx="8" fill="#FEFCBF" stroke="#D69E2E" stroke-width="2"/>
        <text x="515" y="170" font-size="12" font-weight="bold" text-anchor="middle" fill="#744210">Fusion Engine</text>
        <text x="515" y="188" font-size="8.5" fill="#975A16" text-anchor="middle">Dynamic Rescaling</text>
        <text x="515" y="202" font-size="8.5" fill="#975A16" text-anchor="middle">IEEE 7000 Ethics Bias</text>
        <text x="515" y="218" font-size="9" font-weight="bold" fill="#B7791F" text-anchor="middle">0 &ndash; 100% Score</text>

        <!-- Arrow to UI -->
        <path d="M 590 185 L 630 185" stroke="#D69E2E" stroke-width="2" marker-end="url(#arrow-gold)"/>

        <!-- Output Dashboard -->
        <rect x="630" y="125" width="120" height="120" rx="8" fill="#C6F6D5" stroke="#38A169" stroke-width="2"/>
        <text x="690" y="155" font-size="11" font-weight="bold" text-anchor="middle" fill="#22543D">Streamlit UI</text>
        <text x="690" y="175" font-size="8" fill="#2F855A" text-anchor="middle">&bull; Folium GeoMap</text>
        <text x="690" y="190" font-size="8" fill="#2F855A" text-anchor="middle">&bull; Dynamic Gauge</text>
        <text x="690" y="205" font-size="8" fill="#2F855A" text-anchor="middle">&bull; Layer Breakdown</text>
        <text x="690" y="222" font-size="8.5" font-weight="bold" fill="#C53030" text-anchor="middle">Watch/Warn/Alert</text>

        <!-- Markers -->
        <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#718096"/>
            </marker>
            <marker id="arrow-gold" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#D69E2E"/>
            </marker>
        </defs>
    </svg>
</div>

<!-- PAGE 2: LAYER BREAKDOWN -->
<div class="page-break"></div>
<h2>3. In-Depth Breakdown of the 4 Detection Layers</h2>

<div class="grid-2">
    <div class="card">
        <div class="card-header">
            <span>Layer 1: Weather Risk</span>
            <span class="badge badge-blue">Weight: 15%</span>
        </div>
        <p><strong>Question answered:</strong> <em>"Is the forest atmosphere like dry tinder waiting for a spark?"</em></p>
        <p><strong>Algorithm:</strong> Scikit-learn <span class="code-span">RandomForestClassifier</span> trained on meteorological forest fire historical datasets.</p>
        <p><strong>Inputs:</strong></p>
        <ul>
            <li><strong>FFMC:</strong> Moisture in surface litter/leaves.</li>
            <li><strong>DMC / DC:</strong> Moisture in shallow organic soil and deep subsoil layers.</li>
            <li><strong>ISI:</strong> Initial fire spread index (wind + moisture).</li>
            <li><strong>Temp, RH, Wind, Rain:</strong> Ambient weather.</li>
        </ul>
        <p><strong>Output:</strong> Risk score (0.0 to 1.0) and categorical level (Low, Medium, High).</p>
    </div>

    <div class="card">
        <div class="card-header">
            <span>Layer 2: Satellite Thermal</span>
            <span class="badge badge-blue">Weight: 25%</span>
        </div>
        <p><strong>Question answered:</strong> <em>"Do orbiting satellites see thermal hot spots on the ground?"</em></p>
        <p><strong>Data Source:</strong> NASA FIRMS (Fire Information for Resource Management System) VIIRS S-NPP (375m) &amp; MODIS.</p>
        <p><strong>Resilience Features:</strong></p>
        <ul>
            <li><strong>Dual Satellite Queries:</strong> Queries ultra-high resolution VIIRS first; falls back to MODIS if VIIRS is unavailable.</li>
            <li><strong>Offline Resiliency:</strong> Stores successful scans to local cache (<span class="code-span">firms_cache.json</span>) for graceful degradation during internet outages.</li>
            <li><strong>Spatial Deduplication:</strong> Merges hotspot points within 0.01&deg; cells to prevent alert fatigue (IEEE 7010).</li>
        </ul>
    </div>
</div>

<div class="grid-2">
    <div class="card">
        <div class="card-header">
            <span>Layer 3: Vision CNN</span>
            <span class="badge badge-blue">Weight: 30%</span>
        </div>
        <p><strong>Question answered:</strong> <em>"Does an RGB camera physically see flames or smoke plumes?"</em></p>
        <p><strong>Architecture:</strong> Deep Learning Convolutional Neural Network (<span class="code-span">EfficientNetB3</span>) pretrained on wildfire datasets with Sigmoid classification output.</p>
        <p><strong>Pipeline:</strong></p>
        <ul>
            <li>Direct image upload from field cameras/drones.</li>
            <li>Normalizes image into standard 300x300 RGB tensor.</li>
            <li>Runs inference yielding confidence percentage.</li>
            <li><strong>Zero-Crash Fallback:</strong> If TensorFlow is missing, automatically defaults to a color-heuristic computer vision detector (<span class="code-span">DummyVisionDetector</span>).</li>
        </ul>
    </div>

    <div class="card">
        <div class="card-header">
            <span>Layer 4: IoT Ground Sensors</span>
            <span class="badge badge-blue">Weight: 30%</span>
        </div>
        <p><strong>Question answered:</strong> <em>"Are ground sensor nodes detecting acute thermal spikes or smoke?"</em></p>
        <p><strong>Standards:</strong> Conforms to IEEE 1451 (Smart Transducer TEDS) and IEEE 2413 (IoT Architecture).</p>
        <p><strong>Two-Tier Detection Pipeline:</strong></p>
        <ul>
            <li><strong>Rule-Based Trigger:</strong> Immediate alert if Temp &gt; 40&deg;C, RH &lt; 25%, or Smoke &gt; 250 PPM.</li>
            <li><strong>Isolation Forest ML:</strong> Detects multivariate anomalies even if individual metrics haven't crossed single thresholds.</li>
            <li><strong>Staleness Check:</strong> Drops readings older than 5 minutes to prevent acting on historic stale data.</li>
        </ul>
    </div>
</div>

<h2>4. The Fusion Engine &amp; Dynamic Weight Redistribution</h2>
<p>
The core innovation of the Fusion Engine is its ability to handle sensor degradation without generating false conclusions. When all layers are operational, each contributes its baseline weight:
</p>

<div class="diagram-container">
    <svg viewBox="0 0 740 180" xmlns="http://www.w3.org/2000/svg" style="font-family: sans-serif;">
        <!-- Left state: Full -->
        <rect x="20" y="20" width="320" height="140" rx="8" fill="#F7FAFC" stroke="#CBD5E0" stroke-width="1.5"/>
        <text x="180" y="45" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#2B6CB0">Normal State: All 4 Layers Active</text>
        
        <rect x="35" y="65" width="60" height="40" rx="4" fill="#E2E8F0"/><text x="65" y="88" font-size="9" text-anchor="middle" font-weight="bold">Weather</text><text x="65" y="100" font-size="8" text-anchor="middle" fill="#4A5568">15%</text>
        <rect x="105" y="65" width="60" height="40" rx="4" fill="#E2E8F0"/><text x="135" y="88" font-size="9" text-anchor="middle" font-weight="bold">Thermal</text><text x="135" y="100" font-size="8" text-anchor="middle" fill="#4A5568">25%</text>
        <rect x="175" y="65" width="65" height="40" rx="4" fill="#BEE3F8"/><text x="207" y="88" font-size="9" text-anchor="middle" font-weight="bold">Vision</text><text x="207" y="100" font-size="8" text-anchor="middle" fill="#2B6CB0">30%</text>
        <rect x="250" y="65" width="65" height="40" rx="4" fill="#BEE3F8"/><text x="282" y="88" font-size="9" text-anchor="middle" font-weight="bold">IoT</text><text x="282" y="100" font-size="8" text-anchor="middle" fill="#2B6CB0">30%</text>
        <text x="180" y="140" font-size="9" text-anchor="middle" fill="#718096">Total Operational Weight = 100%</text>

        <!-- Arrow -->
        <path d="M 355 90 L 385 90" stroke="#E53E3E" stroke-width="2" marker-end="url(#arrow-red)"/>
        <text x="370" y="80" font-size="8" text-anchor="middle" fill="#E53E3E" font-weight="bold">IoT Fails</text>

        <!-- Right state: Degraded -->
        <rect x="400" y="20" width="320" height="140" rx="8" fill="#FFF5F5" stroke="#FEB2B2" stroke-width="1.5"/>
        <text x="560" y="45" font-size="11.5" font-weight="bold" text-anchor="middle" fill="#9B2C2C">Degraded State: Auto-Redistribution</text>
        
        <rect x="415" y="65" width="65" height="40" rx="4" fill="#FED7D7"/><text x="447" y="88" font-size="9" text-anchor="middle" font-weight="bold">Weather</text><text x="447" y="100" font-size="8.5" text-anchor="middle" fill="#9B2C2C">21.4%</text>
        <rect x="490" y="65" width="65" height="40" rx="4" fill="#FED7D7"/><text x="522" y="88" font-size="9" text-anchor="middle" font-weight="bold">Thermal</text><text x="522" y="100" font-size="8.5" text-anchor="middle" fill="#9B2C2C">35.7%</text>
        <rect x="565" y="65" width="65" height="40" rx="4" fill="#FED7D7"/><text x="597" y="88" font-size="9" text-anchor="middle" font-weight="bold">Vision</text><text x="597" y="100" font-size="8.5" text-anchor="middle" fill="#9B2C2C">42.9%</text>
        <rect x="640" y="65" width="65" height="40" rx="4" fill="#E2E8F0" stroke="#A0AEC0" stroke-dasharray="2 2"/><text x="672" y="88" font-size="9" text-anchor="middle" fill="#A0AEC0">IoT</text><text x="672" y="100" font-size="8" text-anchor="middle" fill="#A0AEC0">Offline</text>
        <text x="560" y="140" font-size="9" text-anchor="middle" fill="#9B2C2C">Proportionally Scaled: 21.4% + 35.7% + 42.9% = 100%</text>

        <defs>
            <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#E53E3E"/>
            </marker>
        </defs>
    </svg>
</div>

<!-- PAGE 3: DECISION MATRIX & COMPLIANCE -->
<div class="page-break"></div>
<h2>5. Ethical Decision Framework &amp; Alert Tiers (IEEE 7000)</h2>
<p>
The system implements an <strong>asymmetric loss function</strong> inspired by IEEE 7000 ethics guidelines. The ethical cost of a <em>false negative</em> (failing to sound an alarm for a real wildfire) is catastrophic: irreversible loss of ancient biodiversity, human casualties, and property devastation. Conversely, a <em>false positive</em> (dispatching a ranger team to verify smoke that turned out to be campfire smoke) carries minimal expense.
</p>

<table>
    <thead>
        <tr>
            <th style="width: 18%;">Alert Level</th>
            <th style="width: 15%;">Score Range</th>
            <th style="width: 32%;">System Interpretation</th>
            <th style="width: 35%;">Action Triggered</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><span class="badge badge-green">WATCH</span></td>
            <td>0.0% &ndash; 40.0%</td>
            <td>Normal baseline environmental conditions. Minor fluctuations in temp or wind.</td>
            <td>Continuous passive background polling. No human intervention needed.</td>
        </tr>
        <tr>
            <td><span class="badge badge-yellow">WARNING</span></td>
            <td>41.0% &ndash; 70.0%</td>
            <td>Elevated danger. Multiple modalities report anomalous metrics (e.g. high thermal + dry air).</td>
            <td>Ranger ground patrols and drone reconnaissance dispatched to sector for visual inspection.</td>
        </tr>
        <tr>
            <td><span class="badge badge-red">CONFIRMED FIRE</span></td>
            <td>71.0% &ndash; 100.0%</td>
            <td>Conclusive multi-modal detection (visual smoke/flame + thermal anomaly + IoT spike).</td>
            <td>Full emergency protocol: automatic siren broadcast, firefighting unit dispatch, aerial suppression alert.</td>
        </tr>
    </tbody>
</table>

<h2>6. IEEE Standards Compliance Matrix</h2>
<p>
Unlike naive machine learning wrappers, this architecture is rigorously designed to satisfy established engineering and ethical standards:
</p>

<table>
    <thead>
        <tr>
            <th style="width: 20%;">Standard</th>
            <th style="width: 35%;">Official Focus</th>
            <th style="width: 45%;">Implementation in This System</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>IEEE 7000</strong></td>
            <td>Model Ethics in Autonomous and AI Systems</td>
            <td>Safety-biased asymmetric alert thresholds. Missing evidence defaults to safe state with transparent reasoning output.</td>
        </tr>
        <tr>
            <td><strong>IEEE 7010</strong></td>
            <td>Wellbeing &amp; Human Operator Impact</td>
            <td>Grid-based spatial deduplication (0.01&deg; radius) prevents alert fatigue and cognitive overload for emergency operators.</td>
        </tr>
        <tr>
            <td><strong>IEEE 2413</strong></td>
            <td>Standard for an Architectural Framework for IoT</td>
            <td>Clear decoupling into Sensing, Network (MQTT), and Application layers with active 300s staleness monitoring.</td>
        </tr>
        <tr>
            <td><strong>IEEE 1451</strong></td>
            <td>Standard for Smart Transducer Interface</td>
            <td>Standardized self-describing sensor metadata payloads (TEDS schema) with uniform engineering units (&deg;C, %, PPM).</td>
        </tr>
    </tbody>
</table>

<h2>7. Key Project Files &amp; Implementation Summary</h2>
<table>
    <thead>
        <tr>
            <th>Module / File</th>
            <th>Primary Class / Function</th>
            <th>Description &amp; Role</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><span class="code-span">app.py</span></td>
            <td>Streamlit Entrypoint</td>
            <td>Renders interactive dashboard, Folium satellite map, live risk gauge, and telemetry inputs.</td>
        </tr>
        <tr>
            <td><span class="code-span">config.py</span></td>
            <td>System Configurations</td>
            <td>Defines bounding boxes for Western Ghats (Sahyadri), presets for Indian wildlife reserves, and weights.</td>
        </tr>
        <tr>
            <td><span class="code-span">fusion/fusion_engine.py</span></td>
            <td><span class="code-span">fuse_scores()</span></td>
            <td>Calculates weighted multi-modal risk score and redistributes weights if any modality is offline.</td>
        </tr>
        <tr>
            <td><span class="code-span">layer1_weather/predict.py</span></td>
            <td><span class="code-span">WeatherRiskPredictor</span></td>
            <td>Random Forest classifier evaluating Canadian Fire Weather Index components.</td>
        </tr>
        <tr>
            <td><span class="code-span">layer2_thermal/firms_client.py</span></td>
            <td><span class="code-span">FIRMSClient</span></td>
            <td>Queries NASA FIRMS VIIRS &amp; MODIS APIs with automatic disk caching and spatial deduplication.</td>
        </tr>
        <tr>
            <td><span class="code-span">layer3_vision/detect.py</span></td>
            <td><span class="code-span">KerasVisionDetector</span></td>
            <td>Deep CNN (EfficientNetB3) fire classifier with <span class="code-span">DummyVisionDetector</span> fallback.</td>
        </tr>
        <tr>
            <td><span class="code-span">layer4_iot/sensor_pipeline.py</span></td>
            <td><span class="code-span">IoTSensorPipeline</span></td>
            <td>IEEE 1451 transducer formatter, physical thresholds, and Isolation Forest anomaly detector.</td>
        </tr>
    </tbody>
</table>

<div style="margin-top: 25px; text-align: center; color: #718096; font-size: 8.5pt;">
    Multi-Modal Forest Fire Early-Warning System &bull; Western Maharashtra (Sahyadri Belt) &bull; Generated for Technical Documentation
</div>

</body>
</html>
"""

html_path = os.path.abspath("Forest_Fire_System_Guide.html")
pdf_path = os.path.abspath("Forest_Fire_Prediction_System_Guide.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML written to {html_path}")

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [
    edge_path,
    "--headless",
    "--no-sandbox",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_path}",
    f"file:///{html_path.replace(os.sep, '/')}"
]

print("Rendering PDF with headless Edge...")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
    print(f"SUCCESS: PDF created at {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")
else:
    print("FAILED to create PDF")
