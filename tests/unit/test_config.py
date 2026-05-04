from earningslora.config import Settings, get_settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.base_model.startswith("meta-llama/")
    assert 0 < s.lora_dropout < 1
    assert s.lora_r >= 1
    assert s.lora_alpha >= s.lora_r
    assert s.max_seq_len >= 512
    assert s.eval_holdout_size > 0
    assert s.eval_seed >= 0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("EARNINGSLORA_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    monkeypatch.setenv("EARNINGSLORA_LORA_R", "32")
    s = Settings(_env_file=None)
    assert s.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert s.lora_r == 32


def test_get_settings_lazy():
    # Just confirms the accessor returns a Settings instance.
    assert isinstance(get_settings(), Settings)
