from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import sys

# Ensure project root is in path so we can import backend.service
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.service import BacktestConfig, BacktestResult, run_backtest_job, ALGORITHM_MAP

app = FastAPI(title="Backtest API", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def root():
    """Render the root welcome page with links to UI and API docs."""
    return """
    <html>
        <head><title>Backtest System</title></head>
        <body style='font-family: sans-serif; text-align: center; padding-top: 50px; background: #111; color: #eee;'>
            <h1>Backtest System</h1>
            <p style='margin-top: 20px;'>
                <a href='/ui' style='font-size: 18px; color: #0af; text-decoration: none; border: 1px solid #0af; padding: 10px 20px; border-radius: 6px;'>
                    Open Backtest Console
                </a>
            </p>
            <p style='margin-top: 20px;'>Or visit <a href='/docs' style='color: #aaa;'>/docs</a> to view API documentation</p>
        </body>
    </html>
    """


@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    """Render the backtest console UI page for selecting algorithm, parameters and viewing results."""
    return """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>Backtest Console</title>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <style>
    body { background:#111; color:#eee; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:20px; }
    .container { max-width:1200px; margin:0 auto; }
    .panel { background:#1e1e1e; border:1px solid #333; border-radius:8px; padding:16px; margin-bottom:16px; }
    label { display:block; font-size:12px; color:#aaa; margin-bottom:4px; }
    input, select { width:100%; padding:6px 8px; border-radius:4px; border:1px solid #444; background:#222; color:#eee; box-sizing:border-box; }
    button { width:100%; padding:10px; border:none; border-radius:4px; background:#0af; color:#111; font-weight:bold; cursor:pointer; margin-top:8px; }
    button:disabled { opacity:0.6; cursor:not-allowed; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; }
    .stat-card { background:#181818; border:1px solid #333; border-radius:6px; padding:8px; }
    .stat-label { font-size:12px; color:#888; }
    .stat-value { font-size:18px; font-weight:bold; }
    .error { background:#611; border:1px solid #a33; border-radius:6px; padding:8px; margin-top:8px; display:none; }
    iframe { width:100%; height:600px; border:none; border-radius:6px; background:#000; }
    @media (min-width: 900px) {
      .layout { display:grid; grid-template-columns:360px 1fr; gap:16px; }
    }
  </style>
</head>
<body>
  <div class='container'>
    <h1>Backtest Console</h1>
    <p style='color:#888;'>Select an algorithm and parameters, run the backtest, and view results below.</p>
    <div class='layout'>
      <div class='panel'>
        <h2 style='margin-top:0;'>Configuration</h2>
        <form id='backtest-form'>
          <div>
            <label>Algorithm</label>
            <select id='algorithm' name='algorithm'></select>
          </div>
          <div style='display:flex; gap:8px; margin-top:8px;'>
            <div style='flex:1;'>
              <label>开始日期</label>
              <input type='date' id='start_date' value='2018-01-01'>
            </div>
            <div style='flex:1;'>
              <label>结束日期</label>
              <input type='date' id='end_date' value='2024-12-31'>
            </div>
          </div>
          <div style='margin-top:8px;'>
            <label>初始资金</label>
            <input type='number' id='initial_capital' value='100000'>
          </div>
          <div style='margin-top:8px;'>
            <label>对比标的 (逗号分隔)</label>
            <input type='text' id='benchmark_cols' value='Nasdaq100, GoldIndex, US30Y, US3M'>
          </div>
          <button type='submit' id='run-btn'>运行回测</button>
          <div id='error-msg' class='error'></div>
        </form>
      </div>
      <div>
        <div id='loading' class='panel' style='display:none; text-align:center;'>
          正在计算回测结果，请稍候...
        </div>
        <div id='results-panel' style='display:none;'>
          <div class='panel'>
            <h2 style='margin-top:0;'>统计结果</h2>
            <div id='stats-grid' class='grid'></div>
          </div>
          <div class='panel'>
            <h2 style='margin-top:0;'>绩效曲线</h2>
            <iframe id='plot-frame' src=''></iframe>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    async function loadAlgorithms() {
      try {
        const res = await fetch('/api/algorithms');
        const algos = await res.json();
        const select = document.getElementById('algorithm');
        algos.forEach(a => {
          const opt = document.createElement('option');
          opt.value = a.key;
          opt.textContent = a.label;
          select.appendChild(opt);
        });
      } catch (e) {
        console.error('加载算法列表失败', e);
      }
    }

    async function runBacktest(e) {
      e.preventDefault();
      const runBtn = document.getElementById('run-btn');
      const loading = document.getElementById('loading');
      const resultsPanel = document.getElementById('results-panel');
      const errorMsg = document.getElementById('error-msg');
      const statsGrid = document.getElementById('stats-grid');
      const plotFrame = document.getElementById('plot-frame');

      errorMsg.style.display = 'none';
      errorMsg.textContent = '';
      resultsPanel.style.display = 'none';
      statsGrid.innerHTML = '';
      plotFrame.src = '';

      runBtn.disabled = true;
      loading.style.display = 'block';

      const algorithm = document.getElementById('algorithm').value;
      const startDate = document.getElementById('start_date').value || null;
      const endDate = document.getElementById('end_date').value || null;
      const initialCapital = parseFloat(document.getElementById('initial_capital').value || '0');
      const benchmarks = document.getElementById('benchmark_cols').value
        .split(',').map(s => s.trim()).filter(Boolean);

      const payload = {
        algorithm: algorithm,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        benchmark_cols: benchmarks
      };

      try {
        const res = await fetch('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Backtest failed');
        }

        const data = await res.json();
        const s = data.stats;
        const metrics = [
          { label: 'Total Return', value: (s['Total Return'] * 100).toFixed(2) + '%' },
          { label: 'CAGR', value: (s['CAGR'] * 100).toFixed(2) + '%' },
          { label: 'Sharpe Ratio', value: s['Sharpe Ratio'].toFixed(2) },
          { label: 'Max Drawdown', value: (s['Max Drawdown'] * 100).toFixed(2) + '%' },
          { label: 'Volatility', value: (s['Volatility'] * 100).toFixed(2) + '%' },
          { label: 'Years', value: s['Years'].toFixed(1) }
        ];

        metrics.forEach(m => {
          const card = document.createElement('div');
          card.className = 'stat-card';
          card.innerHTML = `<div class='stat-label'>${m.label}</div><div class='stat-value'>${m.value}</div>`;
          statsGrid.appendChild(card);
        });

        plotFrame.src = data.result_url;
        resultsPanel.style.display = 'block';
      } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.style.display = 'block';
      } finally {
        loading.style.display = 'none';
        runBtn.disabled = false;
      }
    }

    document.getElementById('backtest-form').addEventListener('submit', runBacktest);
    loadAlgorithms();
  </script>
</body>
</html>"""


@app.get("/api/algorithms")
def list_algorithms():
    """返回所有可选算法及其描述，供前端渲染下拉框。"""
    return [
        {"key": k, "label": v["label"], "description": v["description"]}
        for k, v in ALGORITHM_MAP.items()
    ]


@app.post("/api/backtest", response_model=BacktestResult)
def api_backtest(req: BacktestConfig) -> BacktestResult:
    """Run a backtest synchronously and return stats + HTML result path & URL."""
    try:
        return run_backtest_job(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# Mount static files so that generated HTML can be accessed via /results/*
results_dir = os.path.join(project_root, "data_processed")
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
