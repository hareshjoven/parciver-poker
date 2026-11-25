"""scripts/train_blueprint.py's --resume gates (a stand-in engine: no
abstraction is loaded and nothing is solved)."""
import importlib.util
import types
from pathlib import Path

import pytest

pytest.importorskip("parciver_engine")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tb():
    spec = importlib.util.spec_from_file_location("train_blueprint_cli", ROOT / "scripts" / "train_blueprint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resume_refuses_an_old_engine_and_points_a_refused_checkpoint_to_the_one_before(tb, tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoint.bin"
    ckpt.write_bytes(b"state")
    calls, error = [], ["blueprint resume failed: checkpoint has a negative regret (mccfr+ floors them at 0) at k"]

    class Blueprint:
        variant = "mccfr+"  # the engine reports the checkpoint's variant

        @staticmethod
        def resume(abstraction_dir, path, threads):
            calls.append(path)
            raise ValueError(error[0])

    def run():
        tb.run(tmp_path / "run", abstraction_dir=tmp_path, variant="mccfr+", seed=0, chunk=0, checkpoint_every=0,
               save_minutes=0, target_iters=1, max_seconds=1, resume_from=tmp_path)

    # An engine that does not report it is refused before the (long) load.
    monkeypatch.setattr(tb, "engine", types.SimpleNamespace(Blueprint=types.SimpleNamespace(resume=Blueprint.resume)))
    with pytest.raises(SystemExit, match="rebuild it"):
        run()
    assert calls == []
    monkeypatch.setattr(tb, "engine", types.SimpleNamespace(Blueprint=Blueprint))
    with pytest.raises(SystemExit, match="negative regret") as e:
        run()
    assert "before it" not in str(e.value) and calls == [str(ckpt)]
    # A refused checkpoint points to the one each save keeps before it ...
    (tmp_path / "checkpoint.bin.previous").write_bytes(b"older state")
    with pytest.raises(SystemExit, match=r"negative regret .*; the checkpoint before it: --resume .*checkpoint\.bin\.previous$"):
        run()
    # ... but another failure (the abstraction, here) does not.
    error[0] = "no abstraction at the path"
    with pytest.raises(SystemExit, match="no abstraction") as e:
        run()
    assert "before it" not in str(e.value)
    # Nor does a checkpoint of other bucket tables: the one before it is too.
    error[0] = ("blueprint resume failed: checkpoint was trained on a different abstraction "
                "(fingerprint 0000000000000001; these tables: 0000000000000002), or the file is corrupt")
    with pytest.raises(SystemExit, match="different abstraction") as e:
        run()
    assert "before it" not in str(e.value)
