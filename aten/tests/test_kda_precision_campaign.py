from __future__ import annotations

from compiler.aten.models.kda.precision_campaign import CampaignConfig, run_campaign


def test_precision_campaign_uses_real_kimi_state_dimensions_and_finite_results() -> None:
    report = run_campaign(
        CampaignConfig(
            tokens=(8,),
            seeds=(17, 42),
            key_dim=128,
            value_dim=128,
            checkpoint_intervals=(1, 4),
        )
    )
    assert report["scope"]["real_dimensions"] == {
        "key_dim": 128,
        "value_dim": 128,
        "full_model_heads": 96,
    }
    assert len(report["records"]) == 8
    assert all(record["nan_count"] == 0 for record in report["records"])
    assert all(record["inf_count"] == 0 for record in report["records"])
    fp32 = [record for record in report["records"] if record["storage"] == "fp32"]
    assert all(record["output"]["relative_l2_max"] == 0 for record in fp32)
    assert all(record["state"]["relative_l2_max"] == 0 for record in fp32)


def test_token_storage_accumulates_more_error_than_chunk_storage() -> None:
    report = run_campaign(
        CampaignConfig(
            tokens=(16,),
            seeds=(17,),
            key_dim=128,
            value_dim=128,
            checkpoint_intervals=(1, 8),
        )
    )
    bf16 = {
        record["schedule"]: record["state"]["relative_l2_mean"]
        for record in report["records"]
        if record["storage"] == "bf16"
    }
    assert bf16["token"] >= bf16["chunk8"]
