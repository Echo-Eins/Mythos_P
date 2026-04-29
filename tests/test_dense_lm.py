import torch
import pytest

from open_mythos.dense_lm import DenseLMConfig, OpenMythosDenseLM
from open_mythos.modules import add_loop_index_embedding


def tiny_config() -> DenseLMConfig:
    return DenseLMConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=None,
        prelude_layers=1,
        coda_layers=1,
        max_loop_iters=2,
        max_seq_len=32,
        dropout=0.0,
        eos_token_id=2,
        pad_token_id=0,
    )


def test_forward_loss_shape():
    cfg = tiny_config()
    model = OpenMythosDenseLM(cfg)
    input_ids = torch.randint(3, cfg.vocab_size, (2, 16))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, collect_stats=True)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None
    assert out.stats is not None
    assert not torch.isnan(out.logits).any()
    assert "recurrent_input_mixer_h_gate_mean" in out.stats
    assert "recurrent_output_bridge_h_gate_mean" in out.stats
    assert "recurrent_coda_input_rms" in out.stats
    assert "recurrent_delta_rms" in out.stats


def test_generate_no_cache():
    cfg = tiny_config()
    model = OpenMythosDenseLM(cfg).eval()
    input_ids = torch.randint(3, cfg.vocab_size, (2, 8))
    out = model.generate(
        input_ids,
        max_new_tokens=4,
        n_loops=1,
        do_sample=False,
        eos_token_id=None,
    )
    assert out.shape == (2, 12)


def test_lti_A_is_stable():
    cfg = tiny_config()
    model = OpenMythosDenseLM(cfg)
    A = model.recurrent.injection.A()
    assert A.min().item() > 0.0
    assert A.max().item() < 1.0
    effective_delta = (1.0 - A) * model.recurrent.injection.delta_gain()
    assert effective_delta.abs().mean().item() > 0.1


def test_recurrent_mixers_start_with_configured_gates():
    cfg = tiny_config()
    cfg.recurrent_input_h_init = 0.7
    cfg.recurrent_output_h_init = 0.75
    model = OpenMythosDenseLM(cfg)
    assert torch.allclose(
        model.recurrent.input_mixer.h_gate().mean(),
        torch.tensor(0.7),
        atol=1e-5,
    )
    assert torch.allclose(
        model.recurrent_output_bridge.h_gate().mean(),
        torch.tensor(0.75),
        atol=1e-5,
    )


def test_no_act_config_does_not_allocate_unused_act_parameters():
    model = OpenMythosDenseLM(tiny_config())
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.startswith("recurrent.act.") for name in names)
    assert not any(name.startswith("recurrent.act_norm.") for name in names)


def test_config_rejects_invalid_attention_counts():
    cfg = tiny_config()
    cfg.n_heads = 0
    with pytest.raises(ValueError, match="n_heads"):
        OpenMythosDenseLM(cfg)

    cfg = tiny_config()
    cfg.n_kv_heads = 0
    with pytest.raises(ValueError, match="n_kv_heads"):
        OpenMythosDenseLM(cfg)


def test_loop_index_embedding_matches_legacy_layout():
    h = torch.zeros(1, 1, 8)
    out = add_loop_index_embedding(h, loop_index=0, loop_dim=4)
    assert torch.equal(out[0, 0, :4], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_forward_with_act_stats():
    cfg = tiny_config()
    cfg.use_act = True
    cfg.act_threshold = 0.99
    cfg.act_ponder_weight = 0.01
    model = OpenMythosDenseLM(cfg)
    input_ids = torch.randint(3, cfg.vocab_size, (2, 16))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, collect_stats=True)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None
    assert out.stats is not None
    assert "act_expected_steps" in out.stats
    assert "act_hard_steps" in out.stats
    assert "act_ponder_loss" in out.stats
    assert "act_loss" in out.stats
    assert out.stats["act_expected_steps"].item() >= 1.0


def test_act_min_steps_delays_weighted_output():
    cfg = tiny_config()
    cfg.use_act = True
    cfg.act_min_steps = 2
    model = OpenMythosDenseLM(cfg)
    input_ids = torch.randint(3, cfg.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids.clone(), collect_stats=True)
    assert out.stats is not None
    assert out.stats["act_expected_steps"].item() >= 2.0
