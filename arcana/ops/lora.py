# lora.py -- texture LoRAs as ops: trained on a set, listed with their checkpoints
#
#   lora train ice --set ice --steps 2000 --background
#   lora list
#
# The training itself is lora_train.py. Each checkpoint lands in
# <data dir>/loras/<name>/step-NNNNNN with a sheet drawn at fixed seeds, and the
# texture LoRA method lists it at once.

from __future__ import annotations

from pydantic import BaseModel, Field

from .registry import Args, Context, Result, op, positional

try:
    from .. import boards as _boards
    from .. import lora_train as _train
except ImportError:                                   # pragma: no cover
    import boards as _boards
    import lora_train as _train


class Checkpoint(BaseModel):
    name: str
    step: int | None = None
    sheet: str | None = None
    trigger: str | None = None


class LoraRun(BaseModel):
    name: str
    checkpoints: list[Checkpoint]


class LoraListResult(Result):
    runs: list[LoraRun]

    def text(self) -> str:
        if not self.runs:
            return "No LoRAs trained in Arcana yet. `lora train ice --set ice` trains one on a set.\n"
        return "".join(f"{r.name}: {', '.join(c.name.rsplit('/', 1)[-1] for c in r.checkpoints)}"
                       f"  ·  trigger \"{r.checkpoints[-1].trigger or ''}\"\n" for r in self.runs)


class LoraListArgs(Args):
    pass


@op("lora list", "read", LoraListArgs, LoraListResult,
    doc="""
    Every LoRA trained in Arcana, with its checkpoints. Each checkpoint has a
    sheet drawn at fixed seeds. The texture LoRA method lists them as <name>/step-NNNNNN,
    beside LoRAs found in ARCANA_LORA_DIRS and ~/latent-lab-models.
    """,
    examples=("lora list",))
def _lora_list(a: LoraListArgs, ctx: Context) -> LoraListResult:
    return LoraListResult(runs=[LoraRun(name=r["name"], checkpoints=[Checkpoint(**{k: c[k] for k in
                                ("name", "step", "sheet", "trigger")}) for c in r["checkpoints"]])
                                for r in _train.runs()])


class LoraTrainArgs(Args):
    name: str = positional(0, description="The LoRA's name; its trigger becomes \"in <name> style\".")
    set: str = Field(description="The set it trains on.")
    steps: int = Field(2000, ge=100, le=20000, description="Training steps. Checkpoints every --every steps, "
                       "so a run that goes past its best keeps the earlier ones.")
    every: int = Field(500, ge=50, le=5000, description="Keep a checkpoint, with its sheet, this often.")
    rank: int = Field(16, ge=2, le=128, description="LoRA rank: higher holds more and overfits sooner.")
    lr: float = Field(1e-4, gt=0, le=1e-2, description="Learning rate.")
    detail: float = Field(3.0, ge=1.0, le=8.0, description="How much of a picture a training crop sees: "
                          "3 is a third of its short side, which keeps a photograph's grain.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


class LoraTrainResult(Result):
    name: str
    checkpoints: list[str]
    trigger: str
    pictures: int
    seconds: int
    stopped: bool = False

    def text(self) -> str:
        head = "Stopped early" if self.stopped else "Trained"
        return (f"{head}: {self.name} on {self.pictures} pictures in {self.seconds // 60} min. "
                f"Checkpoints: {', '.join(self.checkpoints) or 'none'}. Prompt with \"{self.trigger}\".\n")


@op("lora train", "effect", LoraTrainArgs, LoraTrainResult,
    doc="""
    Train a texture LoRA on a set's pictures. It trains on Stable Diffusion 1.5,
    the base the texture LoRA method draws on. Each picture is captioned with words from
    Arcana's vocabulary and the trigger "in <name> style". Needs an NVIDIA GPU;
    2000 steps take minutes, not hours. Pick the checkpoint by its sheet: the
    loss does not say when a LoRA starts to blur.
    """,
    examples=("lora train ice --set ice --background", "lora train ice --set ice --steps 3000 --every 500"))
def _lora_train(a: LoraTrainArgs, ctx: Context) -> Result:
    from .effects import run_effect

    rec = _boards.load(a.set)
    if rec is None:
        raise ValueError(f"No set called `{a.set}`.")
    paths, _ = _boards.present(rec)

    def compute(progress):
        cancelled = getattr(progress, "cancelled", None)
        return _train.train(a.name, paths, steps=a.steps, rank=a.rank, lr=a.lr, detail=a.detail, every=a.every,
                            origin={"set": a.set}, progress=progress, cancelled=cancelled)

    def record(tx, payload):
        return LoraTrainResult(name=payload["name"], checkpoints=payload["checkpoints"], trigger=payload["trigger"],
                               pictures=payload["pictures"], seconds=payload["seconds"],
                               stopped=payload["stopped"])

    return run_effect(ctx, f"lora train {a.name}", compute, record, a.background, kind="lora")
