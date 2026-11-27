"""Build and audit the two-class initialization using Ultralytics' native load path."""
import hashlib
import json

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import SemanticSegmentationModel

from .assets import official_weights, verify_weights
from .data import digest

MODEL_YAML = "yolo26s-sem.yaml"
PARAMETERS = 6_498_756


def resolve_initial_source(args, config):
    if not config["pretrained"]:
        if args.weights:
            raise ValueError("Random initialization cannot accept checkpoint weights")
        return MODEL_YAML
    path = args.weights or official_weights(args.data_root / "yolo_weights" / config["weights"])
    if path.name != config["weights"]:
        raise ValueError("Checkpoint does not belong to this experiment")
    verify_weights(path)
    return path


def signature(model):
    layers = [(type(layer).__name__, layer.f) for layer in model.model]
    shapes = [(name, list(value.shape)) for name, value in model.state_dict().items()]
    graph = {key: model.yaml[key] for key in ("backbone", "head", "scale", "nc", "channels")}
    payload = dict(layers=layers, shapes=shapes, graph=graph)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def state_hash(model):
    sha = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        sha.update(name.encode("utf-8") + b"\0")
        sha.update(str(array.dtype).encode("ascii") + b"\0")
        sha.update(array.tobytes())
    return sha.hexdigest()


def initialize(cfg, source, seed, verbose=False):
    # Ultralytics updates a shared YAML dict's nc when constructing the target model.
    source_classes = source.yaml["nc"] if source is not None else None
    torch.manual_seed(seed)
    model = SemanticSegmentationModel(cfg, nc=2, ch=3, verbose=verbose)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != PARAMETERS or model.yaml["scale"] != "s":
        raise ValueError("Expected the Exp01 YOLO26s-sem parameter count and scale")
    target = model.state_dict()
    ignored = {}
    transferred = []
    if source is not None:
        if source.yaml["scale"] != "s" or any(source.yaml[k] != model.yaml[k] for k in ("backbone", "head", "channels")):
            raise ValueError("Checkpoint architecture differs from YOLO26s-sem")
        source_tensors = source.state_dict()
        for name, tensor in target.items():
            previous = source_tensors.get(name)
            if previous is None or tensor.shape != previous.shape:
                ignored[name] = dict(reason="missing" if previous is None else "shape mismatch",
                                     target_shape=list(tensor.shape),
                                     source_shape=list(previous.shape) if previous is not None else None)
            else:
                transferred.append(name)
        # Compare the initialized tensors after the native transfer, including the output layers.
        fresh = {name: target[name].clone() for name in ignored}
        model.load(source, verbose=verbose)
        loaded = model.state_dict()
        if any(not torch.equal(loaded[name], source_tensors[name].to(loaded[name])) for name in transferred):
            raise RuntimeError("Native weight transfer disagrees with the expected matching tensors")
        if any(not torch.equal(loaded[name], value) for name, value in fresh.items()):
            raise RuntimeError("Native weight transfer overwrote an incompatible output tensor")
    else:
        ignored = {name: {"reason": "native random initialization"} for name in target}
    metadata = dict(mode="pretrained" if source is not None else "random", seed=seed,
                    architecture=MODEL_YAML, architecture_sha256=signature(model),
                    initial_state_sha256=state_hash(model), parameters=parameters,
                    target_items=len(target), source_classes=source_classes,
                    transferred_items=len(transferred), transferred_keys=transferred, not_transferred=ignored)
    return model, metadata


def inspect_initialization(args, config):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    asset = resolve_initial_source(args, config)
    source = YOLO(asset).model if config["pretrained"] else None
    model, report = initialize(source.yaml if source is not None else MODEL_YAML, source, config["seed"])
    with torch.inference_mode():
        logits = model.eval()(torch.zeros(1, 3, config["height"], config["width"]))
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
    if logits.ndim != 4 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Unexpected model output from CPU initialization")
    return {**report, "experiment": config["experiment"], "logits_shape": list(logits.shape),
            "source_file": asset.name if config["pretrained"] else None,
            "source_sha256": digest(asset) if config["pretrained"] else None}
