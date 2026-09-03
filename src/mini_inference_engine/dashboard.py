from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mini-Together Inference Engine</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
    }
    .title-group h1 { font-size: 24px; font-weight: 700; color: #fff; }
    .title-group p { font-size: 14px; color: var(--text-muted); }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }

    .grid { display: grid; gap: 16px; margin-bottom: 24px; }
    .grid-5 { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .grid-4 { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .grid-2 { grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px;
    }
    .card-title {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .card-value {
      font-size: 24px;
      font-weight: 700;
      color: #fff;
    }
    .card-subtext {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }

    .worker-card {
      background: #131d31;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
      margin-top: 12px;
    }
    .worker-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .worker-id { font-weight: 600; font-size: 15px; color: var(--accent); }
    .worker-stats {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      text-align: center;
    }
    .stat-box {
      background: rgba(0,0,0,0.2);
      padding: 8px;
      border-radius: 6px;
    }
    .stat-box .lbl { font-size: 11px; color: var(--text-muted); }
    .stat-box .val { font-size: 16px; font-weight: 700; }

    .progress-bar-bg {
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      height: 10px;
      overflow: hidden;
      margin: 8px 0;
    }
    .progress-bar-fill {
      background: linear-gradient(90deg, var(--primary), var(--accent));
      height: 100%;
      border-radius: 9999px;
      transition: width 0.3s ease;
      width: 0%;
    }

    /* Playground section */
    .playground {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    textarea {
      width: 100%;
      height: 90px;
      background: #080c14;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      color: #fff;
      font-family: inherit;
      font-size: 14px;
      resize: vertical;
    }
    textarea:focus { outline: none; border-color: var(--primary); }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }
    .btn {
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 8px 18px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .presets {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .preset-btn {
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
    }
    .preset-btn:hover { color: #fff; background: rgba(255,255,255,0.12); }
    .output-box {
      background: #080c14;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      min-height: 100px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-word;
      color: #e2e8f0;
      max-height: 320px;
      overflow-y: auto;
    }
    .pulse {
      animation: pulse-animation 1.5s infinite;
    }
    @keyframes pulse-animation {
      0% { opacity: 1; }
      50% { opacity: 0.4; }
      100% { opacity: 1; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>Mini-Together Inference Engine</h1>
        <p>Observable Fault-Tolerant LLM Serving Control Plane</p>
      </div>
      <div>
        <span class="status-badge" id="cluster-status">
          <span class="status-dot"></span> <span id="cluster-status-text">Online</span>
        </span>
      </div>
    </header>

    <!-- Top Key Metrics Grid -->
    <div class="grid grid-5">
      <div class="card">
        <div class="card-title">Active Model</div>
        <div class="card-value" id="val-model" style="font-size: 16px; word-break: break-all;">loading...</div>
        <div class="card-subtext" id="val-device">Device: -</div>
      </div>
      <div class="card">
        <div class="card-title">Throughput</div>
        <div class="card-value" id="val-tps" style="color: var(--accent);">0.0 <span style="font-size: 14px; font-weight: 500; color: var(--text-muted);">tok/s</span></div>
        <div class="card-subtext">Live token rate</div>
      </div>
      <div class="card">
        <div class="card-title">Routing Policy</div>
        <div class="card-value" id="val-policy" style="font-size: 18px;">-</div>
        <div class="card-subtext" id="val-workers-summary">0 Workers Active</div>
      </div>
      <div class="card">
        <div class="card-title">Tokens Generated</div>
        <div class="card-value" id="val-tokens">0</div>
        <div class="card-subtext">Total since server startup</div>
      </div>
      <div class="card">
        <div class="card-title">Total Requests</div>
        <div class="card-value" id="val-requests">0</div>
        <div class="card-subtext" id="val-requests-sub">0 ok</div>
      </div>
    </div>

    <!-- Workers & Cache Grid -->
    <div class="grid grid-2">
      <!-- Worker Schedulers -->
      <div class="card">
        <div class="card-title">Worker Schedulers</div>
        <div id="workers-list">Loading workers telemetry...</div>
      </div>

      <!-- Logical KV Cache -->
      <div class="card">
        <div class="card-title">Logical Paged KV-Cache</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 10px;">
          <span style="font-size: 14px; color: var(--text-muted);">Cache Utilization</span>
          <span style="font-size: 18px; font-weight: 700;" id="cache-util-pct">0%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="cache-util-bar"></div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
          <div class="stat-box">
            <div class="lbl">Fragmentation</div>
            <div class="val" id="cache-frag">0%</div>
          </div>
          <div class="stat-box">
            <div class="lbl">Block Pool</div>
            <div class="val" id="cache-blocks">- / -</div>
          </div>
        </div>
        <div class="card-subtext" style="margin-top: 14px;">
          Logical paged allocation model tracking dynamic sequence block assignments and LRU eviction pressure.
        </div>
      </div>
    </div>

    <!-- Interactive Testing Playground -->
    <div class="card playground">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="card-title">Live Model Playground (SSE Streaming)</div>
        <div class="presets">
          <span style="font-size: 12px; color: var(--text-muted); line-height: 24px;">Presets:</span>
          <button class="preset-btn" onclick="setPrompt('What is 25 * 4? Explain your steps.')">Math</button>
          <button class="preset-btn" onclick="setPrompt('Explain quantum computing in two simple sentences.')">Quantum</button>
          <button class="preset-btn" onclick="setPrompt('Tell a witty joke about software developers.')">Joke</button>
          <button class="preset-btn" onclick="setPrompt('Write a Python function to check if a string is a palindrome.')">Code</button>
        </div>
      </div>
      <textarea id="prompt-input" placeholder="Type a prompt to test inference streaming..."></textarea>
      <div class="controls">
        <button class="btn" id="send-btn" onclick="sendPrompt()">Generate Response</button>
        <label style="font-size: 13px; color: var(--text-muted); display: flex; align-items: center; gap: 6px;">
          Max Tokens:
          <input type="number" id="max-tokens" value="48" min="1" max="256" style="width: 60px; background: #080c14; border: 1px solid var(--card-border); color: #fff; padding: 4px 6px; border-radius: 4px;">
        </label>
        <span id="gen-stats" style="font-size: 13px; color: var(--accent); margin-left: auto;"></span>
      </div>
      <div class="output-box" id="output-stream">Tokens will stream here in real time...</div>
    </div>
  </div>

  <script>
    function setPrompt(text) {
      document.getElementById('prompt-input').value = text;
    }

    async function updateDashboard() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('val-model').textContent = data.model || 'mock';
        document.getElementById('val-device').textContent = 'Device: ' + (data.device || 'auto');
        document.getElementById('val-policy').textContent = data.policy || '-';
        const tps = (data.metrics && data.metrics.tokens_per_sec !== undefined) ? data.metrics.tokens_per_sec.toFixed(1) : '0.0';
        document.getElementById('val-tps').innerHTML = tps + ' <span style="font-size: 14px; font-weight: 500; color: var(--text-muted);">tok/s</span>';
        document.getElementById('val-tokens').textContent = (data.metrics && data.metrics.tokens !== undefined) ? data.metrics.tokens : '0';
        document.getElementById('val-requests').textContent = (data.metrics && data.metrics.requests !== undefined) ? data.metrics.requests : '0';

        const workers = data.workers || [];
        document.getElementById('val-workers-summary').textContent = workers.length + ' Workers Registered';

        let workersHtml = '';
        let totalUtil = 0;
        let totalFrag = 0;
        let freeBlocks = 0;
        let totalBlocks = 0;

        workers.forEach(w => {
          const isHealthy = w.healthy;
          const statusClass = isHealthy ? 'style="color: var(--success);"' : 'style="color: var(--danger);"';
          const statusText = isHealthy ? 'Healthy' : 'Unhealthy';
          workersHtml += `
            <div class="worker-card">
              <div class="worker-header">
                <span class="worker-id">${w.id}</span>
                <span style="font-size: 12px; font-weight: 600;" ${statusClass}>● ${statusText}</span>
              </div>
              <div class="worker-stats">
                <div class="stat-box">
                  <div class="lbl">Active</div>
                  <div class="val" style="${w.active > 0 ? 'color: var(--accent);' : ''}">${w.active}</div>
                </div>
                <div class="stat-box">
                  <div class="lbl">Queue</div>
                  <div class="val" style="${w.queue_depth > 0 ? 'color: var(--warning);' : ''}">${w.queue_depth}</div>
                </div>
                <div class="stat-box">
                  <div class="lbl">Avg Latency</div>
                  <div class="val">${w.latency_ms.toFixed(1)} ms</div>
                </div>
              </div>
            </div>
          `;
          if (w.cache_utilization !== undefined) totalUtil = Math.max(totalUtil, w.cache_utilization);
          if (w.cache_fragmentation !== undefined) totalFrag = Math.max(totalFrag, w.cache_fragmentation);
          if (w.cache_free_blocks !== undefined) freeBlocks = w.cache_free_blocks;
          if (w.cache_total_blocks !== undefined) totalBlocks = w.cache_total_blocks;
        });
        document.getElementById('workers-list').innerHTML = workersHtml || 'No workers available';

        const utilPct = Math.round(totalUtil * 100);
        document.getElementById('cache-util-pct').textContent = utilPct + '%';
        document.getElementById('cache-util-bar').style.width = utilPct + '%';
        document.getElementById('cache-frag').textContent = Math.round(totalFrag * 100) + '%';
        document.getElementById('cache-blocks').textContent = (totalBlocks > 0) ? (totalBlocks - freeBlocks) + ' / ' + totalBlocks : '-';
      } catch (e) {
        console.error('Status fetch error:', e);
      }
    }

    async function sendPrompt() {
      const prompt = document.getElementById('prompt-input').value.trim();
      if (!prompt) return;

      const maxTokens = parseInt(document.getElementById('max-tokens').value, 10) || 32;
      const sendBtn = document.getElementById('send-btn');
      const outputBox = document.getElementById('output-stream');
      const genStats = document.getElementById('gen-stats');

      sendBtn.disabled = true;
      outputBox.textContent = '';
      genStats.textContent = 'Generating...';

      const startTime = performance.now();
      let tokenCount = 0;

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [{ role: 'user', content: prompt }],
            max_tokens: maxTokens,
            stream: true
          })
        });

        if (!response.ok) {
          const errData = await response.json();
          outputBox.textContent = 'Error: ' + (errData.error ? errData.error.message : response.statusText);
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop(); // keep remainder

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') break;

            try {
              const chunk = JSON.parse(dataStr);
              if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content) {
                outputBox.textContent += chunk.choices[0].delta.content;
                outputBox.scrollTop = outputBox.scrollHeight;
                tokenCount++;
              }
            } catch (err) {
              console.error('SSE parse error:', err);
            }
          }
        }
        const totalDurationMs = Math.round(performance.now() - startTime);
        genStats.textContent = `Completed: ${tokenCount} tokens in ${totalDurationMs} ms (${(tokenCount / (totalDurationMs / 1000)).toFixed(1)} tok/s)`;
        updateDashboard();
      } catch (err) {
        outputBox.textContent = 'Network / Stream Error: ' + err.message;
      } finally {
        sendBtn.disabled = false;
      }
    }

    // Initial load and periodic refresh
    updateDashboard();
    setInterval(updateDashboard, 1500);
  </script>
</body>
</html>
"""


def get_dashboard_html() -> str:
    return DASHBOARD_HTML
