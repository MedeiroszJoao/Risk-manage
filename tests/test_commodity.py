from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskmanage.commodity import build_commodity_spot_pnl, save_commodity_spot_case


def _eia_frame() -> pd.DataFrame:
    rows = [
        ("2020-04-16", "wti_cushing_spot", "PET.RWTC.D", 18.27),
        ("2020-04-17", "wti_cushing_spot", "PET.RWTC.D", 18.31),
        ("2020-04-20", "wti_cushing_spot", "PET.RWTC.D", -36.98),
        ("2020-04-21", "wti_cushing_spot", "PET.RWTC.D", 8.91),
        ("2020-04-22", "wti_cushing_spot", "PET.RWTC.D", 13.78),
        ("2020-04-23", "wti_cushing_spot", "PET.RWTC.D", 16.50),
        ("2020-04-16", "brent_europe_spot", "PET.RBRTE.D", 27.82),
        ("2020-04-17", "brent_europe_spot", "PET.RBRTE.D", 28.08),
        ("2020-04-20", "brent_europe_spot", "PET.RBRTE.D", 25.57),
        ("2020-04-21", "brent_europe_spot", "PET.RBRTE.D", 19.33),
        ("2020-04-22", "brent_europe_spot", "PET.RBRTE.D", 20.04),
        ("2020-04-23", "brent_europe_spot", "PET.RBRTE.D", 21.33),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["date", "series", "series_id", "value"],
    )
    frame["units"] = "Dollars per Barrel"
    return frame


def test_historical_commodity_spot_pnl_preserves_negative_wti_and_exact_linear_rtpl():
    pnl, summary = build_commodity_spot_pnl(
        _eia_frame(),
        wti_quantity=1_000,
        brent_quantity=-500,
    )

    first = pnl.iloc[0]
    assert first["wti_pnl"] == pytest.approx(40.0)
    assert first["brent_pnl"] == pytest.approx(-130.0)
    assert first["hpl"] == pytest.approx(-90.0)

    negative_print = pnl.iloc[1]
    assert negative_print["wti_previous"] == pytest.approx(18.31)
    assert negative_print["wti_current"] == pytest.approx(-36.98)
    assert negative_print["wti_pnl"] == pytest.approx(-55_290.0)
    assert negative_print["brent_pnl"] == pytest.approx(1_255.0)
    assert negative_print["hpl"] == pytest.approx(-54_035.0)

    recovery = pnl.iloc[2]
    assert recovery["wti_previous"] == pytest.approx(-36.98)
    assert recovery["wti_current"] == pytest.approx(8.91)
    assert recovery["wti_pnl"] == pytest.approx(45_890.0)
    assert recovery["brent_pnl"] == pytest.approx(3_120.0)
    assert recovery["hpl"] == pytest.approx(49_010.0)

    assert pnl["hpl"].tolist() == pytest.approx(pnl["rtpl"].tolist())
    assert pnl["residual"].tolist() == pytest.approx([0.0] * len(pnl))
    assert summary["pnl"]["negative_wti_observations"] == 1
    assert summary["pnl"]["minimum_wti_spot"] == pytest.approx(-36.98)
    assert summary["source"]["common_observation_dates"] == 6
    assert summary["source"]["unmatched_source_observations"] == 0
    assert summary["historical_risk"]["97.5%"]["ten_observation_horizon"] is None
    assert summary["stress_window"] is None


def test_commodity_spot_case_inner_joins_actual_dates_without_forward_fill():
    frame = _eia_frame()
    frame = frame.loc[
        ~(
            (frame["series"] == "brent_europe_spot")
            & (frame["date"] == "2020-04-21")
        )
    ]

    pnl, summary = build_commodity_spot_pnl(frame, 100.0, -100.0)

    assert summary["source"]["common_observation_dates"] == 5
    assert summary["source"]["wti_observations_available"] == 6
    assert summary["source"]["brent_observations_available"] == 5
    assert "2020-04-21" not in pnl["date"].dt.strftime("%Y-%m-%d").tolist()
    assert len(pnl) == 4


def test_commodity_spot_case_rejects_duplicate_dates_and_wrong_series_ids():
    frame = _eia_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate EIA oil observations"):
        build_commodity_spot_pnl(duplicated)

    wrong_id = frame.copy()
    wrong_id.loc[wrong_id["series"] == "brent_europe_spot", "series_id"] = "PET.WRONG.D"
    with pytest.raises(ValueError, match="must map to PET.RBRTE.D"):
        build_commodity_spot_pnl(wrong_id)


def test_commodity_spot_case_requires_matching_units_and_finite_values():
    frame = _eia_frame()
    frame.loc[frame["series"] == "brent_europe_spot", "units"] = "US cents per Barrel"
    with pytest.raises(ValueError, match="units must match"):
        build_commodity_spot_pnl(frame)

    frame = _eia_frame()
    frame.loc[0, "value"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        build_commodity_spot_pnl(frame)


def test_commodity_spot_case_rejects_nonfinite_quantities_and_bad_confidence():
    frame = _eia_frame()
    with pytest.raises(ValueError, match="quantities must be finite"):
        build_commodity_spot_pnl(frame, wti_quantity=float("nan"))
    with pytest.raises(ValueError, match="confidence levels"):
        build_commodity_spot_pnl(frame, confidence_levels=(0.5,))


def test_commodity_spot_risk_outputs_are_saved_with_reproducible_pnl_hash(tmp_path):
    pnl, summary = build_commodity_spot_pnl(_eia_frame())
    pnl_path = tmp_path / "nested" / "commodity.csv"
    summary_path = tmp_path / "nested" / "risk.json"
    save_commodity_spot_case(pnl, summary, pnl_path, summary_path)

    saved = pd.read_csv(pnl_path)
    assert len(saved) == len(pnl)
    assert summary_path.exists()
    assert len(summary["normalized_pnl_sha256"]) == 64



def test_commodity_spot_case_calculates_ten_observation_risk_and_stress_window():
    dates = pd.bdate_range("2007-01-02", periods=310)
    phase = pd.Series(range(len(dates)), dtype=float).to_numpy()
    wti = 65.0 + 3.5 * np.sin(phase / 8.0)
    brent = 69.0 + 2.7 * np.sin(phase / 8.0 + 0.15)
    # Include the historically possible negative WTI spot observation.
    wti[200] = -36.98
    rows = []
    for dt, value in zip(dates, wti):
        rows.append((dt.date().isoformat(), "wti_cushing_spot", "PET.RWTC.D", value, "Dollars per Barrel"))
    for dt, value in zip(dates, brent):
        rows.append((dt.date().isoformat(), "brent_europe_spot", "PET.RBRTE.D", value, "Dollars per Barrel"))
    frame = pd.DataFrame(rows, columns=["date", "series", "series_id", "value", "units"])

    pnl, summary = build_commodity_spot_pnl(frame)

    assert len(pnl) == 309
    assert summary["historical_risk"]["97.5%"]["ten_observation_horizon"] is not None
    assert summary["historical_risk"]["99.0%"]["ten_observation_horizon"] is not None
    assert summary["stress_window"] is not None
    assert summary["stress_window"]["end_index"] - summary["stress_window"]["start_index"] + 1 == 250
    assert summary["stress_window"]["start_date"] <= summary["stress_window"]["end_date"]
    assert summary["pnl"]["negative_wti_observations"] == 1
