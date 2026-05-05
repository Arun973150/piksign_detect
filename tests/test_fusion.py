from backend.fusion import fuse


def test_ensemble_dominates_when_available():
    result = fuse(
        {
            "ensemble": 0.90,
            "synthid": 0.0,
            "ela": 0.0,
            "noise_residual": 0.0,
            "metadata": 0.0,
        }
    )
    assert result.probability > 0.40
    assert result.weights_used["ensemble"] > result.weights_used["synthid"]


def test_synthid_bonus_applies_for_strong_tier():
    result = fuse(
        {
            "synthid": 0.90,
            "ela": 0.0,
            "noise_residual": 0.0,
            "metadata": 0.0,
        },
        synthid_tier="strong",
    )
    assert result.synthid_bonus == 0.10
    assert result.probability > result.contributions["synthid"]
