"""Tests for Anonymizer and SessionManifest."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.engine.anonymization import Anonymizer
from backtester.store.manifest import SessionManifest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def price_df() -> pd.DataFrame:
    """30-bar synthetic OHLCV frame with a DatetimeIndex."""
    rng    = np.random.default_rng(99)
    dates  = pd.date_range("2022-01-03", periods=30, freq="B")
    closes = 100.0 + np.cumsum(rng.standard_normal(30))
    noise  = rng.uniform(0.1, 0.5, 30)
    return pd.DataFrame(
        {
            "open":   closes,
            "high":   closes + noise,
            "low":    closes - noise,
            "close":  closes,
            "volume": np.ones(30) * 1_000,
        },
        index=dates,
    )


# ── Anonymizer: price ─────────────────────────────────────────────────────────

class TestAnonymizePrices:
    def test_first_close_maps_to_1000(self, price_df):
        """close.iloc[0] after anonymization must equal exactly 1000.0."""
        a = Anonymizer(seed=42)
        anon = a.anonymize_prices(price_df, start_index=0)
        assert abs(anon["close"].iloc[0] - 1000.0) < 1e-9

    def test_arbitrary_start_index(self, price_df):
        """The anchor bar, not bar 0, should map to 1000."""
        a = Anonymizer(seed=42)
        anon = a.anonymize_prices(price_df, start_index=10)
        assert abs(anon["close"].iloc[10] - 1000.0) < 1e-9

    def test_determinism(self, price_df):
        """Two Anonymizers with the same seed must produce identical output."""
        r1 = Anonymizer(seed=7).anonymize_prices(price_df)
        r2 = Anonymizer(seed=7).anonymize_prices(price_df)
        pd.testing.assert_frame_equal(r1[["open", "high", "low", "close"]],
                                       r2[["open", "high", "low", "close"]])

    def test_different_seeds_differ(self, price_df):
        """Different seeds must produce different scale factors."""
        a1 = Anonymizer(seed=1).anonymize_prices(price_df)
        a2 = Anonymizer(seed=2).anonymize_prices(price_df)
        assert not np.allclose(a1["close"].values, a2["close"].values)

    def test_stores_a_and_b(self, price_df):
        """anonymize_prices() must populate self.a and self.b."""
        a = Anonymizer(seed=42)
        a.anonymize_prices(price_df)
        assert a.a is not None and a.b is not None

    def test_a_in_valid_range(self, price_df):
        """a must lie in the seeded Uniform(0.5, 2.0) draw range."""
        a = Anonymizer(seed=42)
        a.anonymize_prices(price_df)
        assert 0.5 <= a.a <= 2.0

    def test_ohlc_consistency_preserved(self, price_df):
        """After anonymization high >= low must still hold for every bar."""
        a    = Anonymizer(seed=42)
        anon = a.anonymize_prices(price_df)
        assert (anon["high"] >= anon["low"]).all()


class TestRestorePrices:
    def test_roundtrip_close(self, price_df):
        """anonymize then restore must recover original close to 6 dp."""
        a    = Anonymizer(seed=42)
        anon = a.anonymize_prices(price_df)
        restored = a.restore_prices(anon)
        np.testing.assert_array_almost_equal(
            restored["close"].values,
            price_df["close"].values,
            decimal=6,
        )

    def test_roundtrip_all_ohlc(self, price_df):
        """Roundtrip must recover all four OHLC columns to 6 dp."""
        a    = Anonymizer(seed=99)
        anon = a.anonymize_prices(price_df)
        restored = a.restore_prices(anon)
        for col in ("open", "high", "low", "close"):
            np.testing.assert_array_almost_equal(
                restored[col].values,
                price_df[col].values,
                decimal=6,
                err_msg=f"Roundtrip failed for column '{col}'",
            )

    def test_restore_without_anonymize_raises(self, price_df):
        """restore_prices() must raise if anonymize_prices() was not called."""
        a = Anonymizer(seed=42)
        with pytest.raises(RuntimeError, match="anonymize_prices"):
            a.restore_prices(price_df)


class TestTransformArray:
    def test_transform_prices_array(self, price_df):
        """transform_prices_array must apply P' = a*P + b element-wise."""
        a    = Anonymizer(seed=42)
        a.anonymize_prices(price_df)
        arr  = price_df["close"].to_numpy()
        result = a.transform_prices_array(arr)
        expected = a.a * arr + a.b
        np.testing.assert_array_almost_equal(result, expected)


# ── Anonymizer: dates ─────────────────────────────────────────────────────────

class TestAnonymizeDates:
    def test_first_date_maps_to_anchor(self, price_df):
        """After anonymize_dates the first bar must land on anchor_date."""
        a    = Anonymizer(anchor_date="1990-01-01")
        anon = a.anonymize_dates(price_df)
        assert str(anon.index[0].date()) == "1990-01-01"

    def test_relative_spacing_preserved(self, price_df):
        """Bar spacing (timedelta between consecutive bars) must be unchanged."""
        a    = Anonymizer()
        anon = a.anonymize_dates(price_df)
        orig_diffs = price_df.index[1:] - price_df.index[:-1]
        anon_diffs = anon.index[1:] - anon.index[:-1]
        assert (orig_diffs == anon_diffs).all()

    def test_stores_time_delta(self, price_df):
        """anonymize_dates() must populate self._time_delta."""
        a = Anonymizer()
        a.anonymize_dates(price_df)
        assert a._time_delta is not None

    def test_get_anonymized_dates_length(self, price_df):
        """get_anonymized_dates must return the same number of dates."""
        a     = Anonymizer()
        a.anonymize_dates(price_df)
        dates = list(price_df.index.to_pydatetime())
        anon  = a.get_anonymized_dates(dates)
        assert len(anon) == len(dates)

    def test_get_anonymized_dates_without_anonymize_raises(self, price_df):
        a = Anonymizer()
        with pytest.raises(RuntimeError, match="anonymize_dates"):
            a.get_anonymized_dates(list(price_df.index.to_pydatetime()))


# ── Anonymizer: to_dict ───────────────────────────────────────────────────────

class TestToDict:
    def test_keys_present(self, price_df):
        a = Anonymizer(seed=5, anchor_date="2000-06-15", preserve_seasonality=False)
        a.anonymize_prices(price_df)
        a.anonymize_dates(price_df)
        d = a.to_dict()
        for key in ("seed", "anchor_date", "preserve_seasonality", "a", "b", "time_delta_days"):
            assert key in d, f"Missing key: {key}"

    def test_values_match(self, price_df):
        a = Anonymizer(seed=17)
        a.anonymize_prices(price_df)
        a.anonymize_dates(price_df)
        d = a.to_dict()
        assert d["seed"] == 17
        assert math.isclose(d["a"], a.a)
        assert math.isclose(d["b"], a.b)
        assert d["time_delta_days"] is not None

    def test_json_serialisable(self, price_df):
        a = Anonymizer(seed=42)
        a.anonymize_prices(price_df)
        a.anonymize_dates(price_df)
        # Must not raise
        json.dumps(a.to_dict())


# ── SessionManifest ───────────────────────────────────────────────────────────

class TestSessionManifest:
    def _make(self, **overrides) -> SessionManifest:
        defaults = dict(
            session_id           = "sess-001",
            symbol               = "AAPL",
            timeframe            = "1D",
            bar_range            = ("2022-01-03", "2022-12-30"),
            data_checksum        = "abcdef1234567890",
            indicator_config     = {"bb_period": 20, "sma_period": 20},
            anonymization_config = None,
            created_at           = "2026-04-13T00:00:00",
        )
        defaults.update(overrides)
        return SessionManifest(**defaults)

    def test_to_json_is_string(self):
        assert isinstance(self._make().to_json(), str)

    def test_roundtrip_all_fields(self):
        """to_json → from_json must recover every field exactly."""
        m  = self._make()
        m2 = SessionManifest.from_json(m.to_json())
        assert m2.session_id           == m.session_id
        assert m2.symbol               == m.symbol
        assert m2.timeframe            == m.timeframe
        assert m2.bar_range            == m.bar_range
        assert m2.data_checksum        == m.data_checksum
        assert m2.indicator_config     == m.indicator_config
        assert m2.anonymization_config == m.anonymization_config
        assert m2.created_at           == m.created_at

    def test_roundtrip_with_anonymization_config(self, price_df):
        """Manifest round-trips correctly when anonymization_config is set."""
        anon = Anonymizer(seed=42)
        anon.anonymize_prices(price_df)
        anon.anonymize_dates(price_df)
        m  = self._make(anonymization_config=anon.to_dict())
        m2 = SessionManifest.from_json(m.to_json())
        assert m2.anonymization_config["seed"] == 42
        assert m2.anonymization_config["a"]    == anon.a

    def test_bar_range_is_tuple(self):
        """bar_range must deserialise back to a tuple, not a list."""
        m  = self._make()
        m2 = SessionManifest.from_json(m.to_json())
        assert isinstance(m2.bar_range, tuple)

    def test_save_and_load(self, tmp_path):
        """save() then load() must recover an identical manifest."""
        m    = self._make()
        path = str(tmp_path / "manifest.json")
        m.save(path)
        m2 = SessionManifest.load(path)
        assert m2.session_id == m.session_id
        assert m2.bar_range  == m.bar_range

    def test_json_contains_all_keys(self):
        """Serialised JSON must contain all expected top-level keys."""
        payload = json.loads(self._make().to_json())
        for key in (
            "session_id", "symbol", "timeframe", "bar_range",
            "data_checksum", "indicator_config",
            "anonymization_config", "created_at",
        ):
            assert key in payload, f"Missing JSON key: {key}"
