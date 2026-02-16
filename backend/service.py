import os
import sys
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

# Ensure project root is in path so we can import strategies
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from strategies.backtest_engine import PermanentPortfolioStrategy
from strategies.algorithms import permanent_portfolio_rebalance

# Algorithm mapping table: manages all available strategies
ALGORITHM_MAP = {
    "permanent_portfolio": {
        "fn": permanent_portfolio_rebalance,
        "label": "Permanent Portfolio (Equal Weight)",
        "description": "Standard equal-weight rebalance strategy (equally allocates across all configured assets)",
    },
    # Future algorithms can be added here
}


class BacktestConfig(BaseModel):
    data_file: str = Field(
        default="data_processed/aligned_assets.csv",
        description="Path to aligned data CSV (relative to project root)",
    )
    output_dir: str = Field(
        default="data_processed",
        description="Directory to save backtest HTML results",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Backtest start date 'YYYY-MM-DD'. None = from beginning.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Backtest end date 'YYYY-MM-DD'. None = to latest.",
    )
    initial_capital: float = Field(
        default=100_000.0,
        description="Initial portfolio capital",
    )
    fees: float = Field(
        default=0.0005,
        description="Transaction fee rate (e.g. 0.0005 = 5bps)",
    )
    rebalance_freq: str = Field(
        default="QE",
        description="Rebalance frequency: 'QE' (Quarter End), 'YE' (Year End)",
    )
    benchmark_cols: List[str] = Field(
        default_factory=lambda: ["Nasdaq100", "GoldIndex", "US30Y", "US3M"],
        description="Columns in the aligned data to show as benchmarks in the plot",
    )
    algorithm: str = Field(
        default="permanent_portfolio",
        description="Algorithm key selected by front-end",
    )


class BacktestResult(BaseModel):
    stats: Dict[str, Any]
    result_html_path: str
    result_url: str
    algorithm: str  # Echo the selected algorithm key


def _resolve_path(path: str) -> str:
    """Resolve a path relative to the project root."""
    return os.path.join(project_root, path)


def run_backtest_job(cfg: BacktestConfig) -> BacktestResult:
    """Core backtest execution logic shared by CLI and API."""
    data_file_abs = _resolve_path(cfg.data_file)
    output_dir_abs = _resolve_path(cfg.output_dir)

    if not os.path.exists(data_file_abs):
        raise FileNotFoundError(f"Data file not found: {cfg.data_file}")

    os.makedirs(output_dir_abs, exist_ok=True)

    # Determine rebalance algorithm
    algo_info = ALGORITHM_MAP.get(cfg.algorithm)
    if not algo_info:
        raise RuntimeError(f"Unsupported algorithm: {cfg.algorithm}")
    
    rebalance_fn = algo_info["fn"]

    # Initialize and run strategy
    strategy = PermanentPortfolioStrategy(data_file_abs, cfg.initial_capital)
    strategy.run_backtest(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        rebalance_freq=cfg.rebalance_freq,
        fees=cfg.fees,
        rebalance_fn=rebalance_fn
    )

    stats = strategy.get_performance_stats()
    if not stats:
        raise RuntimeError("No stats generated. Check date range or data.")

    html_path = strategy.plot_results(
        output_dir=output_dir_abs,
        benchmark_cols=cfg.benchmark_cols,
    )
    if not html_path:
        raise RuntimeError("Failed to generate backtest plot.")

    rel_html_path = os.path.relpath(html_path, project_root)
    rel_inside_output = os.path.relpath(html_path, output_dir_abs)
    result_url = f"/results/{rel_inside_output.replace(os.sep, '/')}"

    return BacktestResult(
        stats=stats, 
        result_html_path=rel_html_path, 
        result_url=result_url,
        algorithm=cfg.algorithm
    )
