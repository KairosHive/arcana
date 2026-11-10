"""The models a mask can ask what a picture shows: its main subject, a named thing, a kind of stuff,
and how far each place is.

- subject   BiRefNet (MIT): a soft matte of whatever the picture is of.
- thing     Grounding DINO (Apache) finds each named thing as a box, SAM 2.1 (Apache) cuts it out.
- stuff     CLIPSeg (Apache): a soft map of what words name, surfaces and qualities included
            ("sky", "wet", "rust"), which an object detector does not see as things.
- depth     DPT hybrid MiDaS (Apache): relative depth, near high.

Each model loads on first use under the memory manager (memory.py) and is let go when memory runs
short. Answers are kept per picture, so moving a mask's other knobs costs nothing after the first.
Ported from latent-lab's segment.py and explore.py.
"""

try:
    from arcana import hub as _hub
except ImportError:                                   # pragma: no cover
    import hub as _hub
import collections
import threading

import numpy as np
import torch
from PIL import Image

from .memory import LOAD, MODELS

SAM = "facebook/sam2.1-hiera-small"
DINO = "IDEA-Research/grounding-dino-tiny"
CLIPSEG = "CIDAS/clipseg-rd64-refined"
SUBJECT = "ZhengPeng7/BiRefNet_dynamic-matting"
DEPTH = "Intel/dpt-hybrid-midas"
KEEP = 48
# What each model took when loaded in latent-lab, in GB of memory, so a load is not refused for room it
# does not need.
GUESS = {"sam": 0.5, "dino": 0.7, "clipseg": 0.6, "subject": 1.0, "depth": 0.6}


def words_of(text):
    """Phrases separated by commas or "and": each is looked for on its own."""
    parts = [p.strip() for chunk in str(text or "").split(",") for p in chunk.split(" and ")]
    return [p for p in parts if p]


class Vision:
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.lock = threading.Lock()
        self.memo = collections.OrderedDict()
        self.embeddings = collections.OrderedDict()

    # ---- what a mask asks ----
    def ask(self, key, picture, kind, arg=""):
        """The map for `kind` on the picture named `key` (made by `picture()`, a PIL image, on a miss),
        at the picture's size, in 0..1. `kind` is subject, thing, stuff or depth."""
        k = (key, kind, str(arg))
        with self.lock:
            if k in self.memo:
                self.memo.move_to_end(k)
                return self.memo[k]
            image = picture()
            with torch.no_grad():
                if kind == "subject":
                    out = self.subject(image)
                elif kind == "thing":
                    out = self.thing(key, image, arg)
                elif kind == "stuff":
                    out = self.stuff(image, arg)
                elif kind == "depth":
                    out = self.depth(image)
                else:
                    raise ValueError(f"no model mask called {kind!r}")
            self.memo[k] = np.asarray(out, np.float32)
            while len(self.memo) > KEEP:
                self.memo.popitem(last=False)
            return self.memo[k]

    # ---- the models, each loaded once and let go when memory is short ----
    def model(self, name):
        with LOAD:
            if name not in self.models:
                with MODELS.loading("mask: " + name, lambda: self.models.pop(name, None), self.lock,
                                    guess=GUESS.get(name, 0.8)):
                    self.models[name] = self._load(name)
                print(f"  alchemy: mask model {name} up", flush=True)
            got = self.models[name]
        MODELS.used("mask: " + name)
        return got

    def _load(self, name):
        dev = self.device
        if name == "sam":
            from transformers import Sam2Model, Sam2Processor

            return _hub.pretrained(Sam2Processor.from_pretrained, SAM), _hub.pretrained(Sam2Model.from_pretrained, SAM).to(dev).eval()
        if name == "dino":
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            return _hub.pretrained(AutoProcessor.from_pretrained, DINO), _hub.pretrained(AutoModelForZeroShotObjectDetection.from_pretrained, DINO).to(dev).eval()
        if name == "clipseg":
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

            return _hub.pretrained(CLIPSegProcessor.from_pretrained, CLIPSEG), _hub.pretrained(CLIPSegForImageSegmentation.from_pretrained, CLIPSEG).to(dev).eval()
        if name == "subject":
            from transformers import AutoModelForImageSegmentation

            model = _hub.pretrained(AutoModelForImageSegmentation.from_pretrained, SUBJECT, trust_remote_code=True)
            return model.to(dev).eval().half() if str(dev).startswith("cuda") else model.to(dev).eval()
        if name == "depth":
            from transformers import DPTForDepthEstimation, DPTImageProcessor

            return _hub.pretrained(DPTImageProcessor.from_pretrained, DEPTH), _hub.pretrained(DPTForDepthEstimation.from_pretrained, DEPTH).to(dev).eval()
        raise ValueError(name)

    # ---- each kind ----
    def subject(self, image, side=1024):
        """BiRefNet's matte of the main subject, read at about 1024 px in the picture's shape."""
        model = self.model("subject")
        w, h = image.size
        k = side / max(w, h)
        sw, sh = max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)
        x = torch.from_numpy(np.asarray(image.convert("RGB").resize((sw, sh), Image.BILINEAR), np.float32) / 255.0)
        x = (x - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        p = next(model.parameters())
        got = model(x.permute(2, 0, 1)[None].to(p.device, p.dtype))[-1].sigmoid().float()
        return torch.nn.functional.interpolate(got, size=(h, w), mode="bilinear")[0, 0].cpu().numpy()

    def sam_embedding(self, key, image):
        processor, model = self.model("sam")
        if key not in self.embeddings:
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            self.embeddings[key] = model.get_image_embeddings(inputs["pixel_values"])
            while len(self.embeddings) > 6:
                self.embeddings.popitem(last=False)
        self.embeddings.move_to_end(key)
        return self.embeddings[key]

    def sam(self, key, image, multimask, **prompts):
        processor, model = self.model("sam")
        inputs = processor(images=image, return_tensors="pt", **prompts).to(self.device)
        out = model(image_embeddings=self.sam_embedding(key, image), multimask_output=multimask,
                    **{k: inputs[k] for k in ("input_points", "input_labels", "input_boxes") if k in inputs})
        logits = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=False)[0]
        best = out.iou_scores[0].argmax(-1).cpu()
        chosen = logits[torch.arange(logits.shape[0]), best]
        return torch.sigmoid(chosen.float()).max(0).values.numpy()

    def thing(self, key, image, text):
        """Grounding DINO's boxes for each phrase, each cut out by SAM 2, all of them together. An empty
        map when nothing named is found."""
        labels = words_of(text)
        w, h = image.size
        if not labels:
            return np.zeros((h, w), np.float32)
        processor, model = self.model("dino")
        inputs = processor(images=image, text=[labels], return_tensors="pt").to(self.device)
        found = processor.post_process_grounded_object_detection(
            model(**inputs), inputs["input_ids"], threshold=0.3, text_threshold=0.25, target_sizes=[(h, w)])[0]
        boxes = found["boxes"].cpu().tolist()
        if not boxes:
            return np.zeros((h, w), np.float32)
        return self.sam(key, image, multimask=False, input_boxes=[boxes])

    def stuff(self, image, text):
        """CLIPSeg's soft map for each phrase (it sees a 352 px square), the strongest phrase kept."""
        labels = words_of(text)
        w, h = image.size
        if not labels:
            return np.zeros((h, w), np.float32)
        processor, model = self.model("clipseg")
        inputs = processor(text=labels, images=[image] * len(labels), padding=True, return_tensors="pt").to(self.device)
        logits = model(**inputs).logits
        logits = logits.reshape(len(labels), *logits.shape[-2:])
        got = torch.sigmoid(logits).max(0).values.float().cpu().numpy()
        return np.asarray(Image.fromarray(got, "F").resize((w, h), Image.BILINEAR))

    def depth(self, image):
        """Relative depth in 0..1, near high."""
        processor, net = self.model("depth")
        w, h = image.size
        out = net(**processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)).predicted_depth
        d = torch.nn.functional.interpolate(out[None], size=(h, w), mode="bicubic")[0, 0]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        return d.float().cpu().numpy()


_VISION = None
_VLOCK = threading.Lock()


def vision():
    """The one Vision this process uses."""
    global _VISION
    with _VLOCK:
        if _VISION is None:
            from .spaces import device

            _VISION = Vision(device())
        return _VISION
