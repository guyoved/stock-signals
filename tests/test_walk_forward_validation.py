import pandas as pd

from models.trainer import walk_forward_validate


def _make_price_df():
    rows = []
    price = 100.0
    for i in range(220):
        price = price * (1 + (0.001 if i % 5 else -0.002))
        rows.append(
            {
                "Open": price * 0.998,
                "High": price * 1.01,
                "Low": price * 0.99,
                "Close": price,
                "Volume": 1000000 + i * 1000,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_validate_returns_metrics():
    df = _make_price_df()
    result = walk_forward_validate(df, n_splits=3, horizon=5, min_confidence=0.55)

    assert "accuracy" in result
    assert "win_rate" in result
    assert "avg_return_pct" in result
    assert "n_trades" in result
    assert result["n_trades"] >= 0
