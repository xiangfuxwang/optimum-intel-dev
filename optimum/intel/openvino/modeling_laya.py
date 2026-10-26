#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings

from .configuration import OVQuantizationConfigBase
from .modeling import MODEL_START_DOCSTRING, OVModel


# Question types understood by the Laya decision head.
QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

LAYA_CONFIG_NAME = "rl_agent_config.json"


class LayaDecisionModel(nn.Module):
    """Laya decision model: bidirectional encoder (ModernBERT) + from-scratch decision head.

    Each option gets a ``[MASK]`` marker token; the head scores the markers and a softmax over the
    question's options gives calibrated probabilities. This is a faithful re-implementation of the
    ``DecisionModel`` released alongside the Convai Laya checkpoints (Apache 2.0), kept here so the
    OpenVINO export does not depend on repository-specific Python files.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # act head sees the pooled sequence + detached summary of its own answer distribution
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _laya_config_from_dict(config_dict: Dict[str, Any], name_or_path: Optional[str] = None) -> PretrainedConfig:
    """Build a `PretrainedConfig` for a Laya checkpoint from its rl_agent_config.json."""
    from transformers import GenerationConfig

    config = PretrainedConfig()
    config.model_type = "laya"
    config.name_or_path = name_or_path
    # Keys colliding with generation parameter names (e.g. `temperature`, which for Laya is a
    # per-question-type softmax calibration constant, not a sampling parameter) are kept out of
    # the config attributes: transformers refuses to save configs carrying generation parameters.
    # They are preserved in rl_agent_config.json instead.
    generation_param_names = set(GenerationConfig().to_dict().keys())
    for key, value in config_dict.items():
        if key in generation_param_names:
            continue
        setattr(config, key, value)
    return config


class _LayaForDecision:
    """Wrapper for loading a Laya decision model (Convai System 1) with a config conforming to
    optimum-intel expectations.

    Laya checkpoints are plain PyTorch checkpoints (``model.safetensors`` + ``rl_agent_config.json`` +
    a Hugging Face encoder under ``encoder/``) rather than ``transformers`` models, so this loader
    re-builds the ``LayaDecisionModel`` architecture, loads the weights and tags the result for the
    OpenVINO export pipeline.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        **kwargs,
    ):
        from safetensors.torch import load_file

        model_path = Path(model_name_or_path)
        is_local_dir = model_path.is_dir()

        def _get_file(filename: str) -> Path:
            if is_local_dir:
                file_path = model_path / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"{filename} not found in {model_path}")
                return file_path
            return Path(
                hf_hub_download(
                    repo_id=str(model_name_or_path),
                    filename=filename,
                    cache_dir=cache_dir,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            )

        config_file = _get_file(LAYA_CONFIG_NAME)
        encoder_config_file = _get_file("encoder/config.json")
        weights_file = _get_file("model.safetensors")

        with open(config_file, "r", encoding="utf-8") as f:
            laya_config_dict = json.load(f)

        encoder_config = AutoConfig.from_pretrained(str(encoder_config_file.parent))
        # eager attention keeps the traced graph fully decomposed for the OpenVINO PyTorch frontend
        encoder = AutoModel.from_config(encoder_config, attn_implementation="eager")
        n_act = len(laya_config_dict.get("act_costs", [])) + 1
        model = LayaDecisionModel(
            encoder,
            head_layers=laya_config_dict.get("head_layers", 2),
            n_act=n_act,
            dropout=laya_config_dict.get("dropout", 0.1),
        )
        state_dict = load_file(str(weights_file))
        model.load_state_dict(state_dict, strict=True)
        model.to("cpu").eval()

        model._laya_model = True
        model._laya_repo_id = str(model_name_or_path)

        config = _laya_config_from_dict(laya_config_dict, name_or_path=str(model_name_or_path))
        # expose encoder properties used by the dummy input generator
        config.vocab_size = getattr(encoder_config, "vocab_size", 50368)
        config.hidden_size = getattr(encoder_config, "hidden_size", 1024)
        model.config = config

        return model


# ---------------------------------------------------------------------------#
# Inference helpers (mirrors the official rl_agent_api.py / rl_common.py logic)
# ---------------------------------------------------------------------------#
def _serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _render_options(q: Dict) -> List[str]:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


def _build_sequence(tok, state, q: Dict, max_len: int, head_max_len: int):
    """[CLS] <type> instructions [SEP] <MASK> opt0 <MASK> opt1 ... [SEP] state [SEP].

    Returns input_ids and the positions of the per-option ``[MASK]`` markers.
    """
    mask_tok = tok.mask_token
    opts = _render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for opt in opts:
        opt_ids.append([tok.mask_token_id] + tok(" " + opt.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(_serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def _collate_items(items: List[Dict], pad_id: int):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
    qtype = torch.tensor([it["qtype"] for it in items], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "qtype": qtype}


def _confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def _temp_bucket(qtype: int, k: int) -> str:
    """Key for per-cardinality temperature fitting."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


@add_start_docstrings(
    """
    OpenVINO Model for Laya decision models (Convai System 1).
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForLayaDecision(OVModel):
    """
    OpenVINO runtime for Laya decision models. The exported IR takes
    ``(input_ids, attention_mask, marker_pos, marker_mask, qtype)`` and returns
    ``(logits, act_logits)``; this class wraps it with the official `system_one` API so a converted
    model can be a drop-in replacement for the PyTorch original.

    Example usage:
    ```python
    >>> from optimum.intel.openvino.modeling_laya import OVModelForLayaDecision

    >>> model = OVModelForLayaDecision.from_pretrained("Convai/laya-en", export=True)
    >>> result = model.system_one(state, questions)
    ```
    """

    export_feature = "feature-extraction"
    _library_name = "laya"

    def _reshape(self, model, batch_size, sequence_length, height=None, width=None):
        # Laya has inputs with different ranks (qtype is 1D), so only reshape
        # dimensions that exist in each input.
        shapes = {}
        for inp in model.inputs:
            shape = inp.get_partial_shape()
            if len(shape) >= 1:
                shape[0] = batch_size
            if len(shape) >= 2:
                shape[1] = sequence_length
            shapes[inp] = shape
        model.reshape(shapes)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Optional[PretrainedConfig] = None,
        export: bool = False,
        force_download: bool = False,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Build the Laya config ourselves: `rl_agent_config.json` has no transformers counterpart,
        so the generic `AutoConfig` lookup would fail on the unknown `laya` model type."""
        if config is None:
            laya_config_dict = {}
            model_path = Path(model_id)
            if model_path.is_dir():
                config_file = model_path / LAYA_CONFIG_NAME
                if config_file.exists():
                    with open(config_file, "r", encoding="utf-8") as f:
                        laya_config_dict = json.load(f)
            else:
                try:
                    config_file = hf_hub_download(
                        repo_id=str(model_id),
                        filename=LAYA_CONFIG_NAME,
                        cache_dir=cache_dir,
                        token=token,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                    with open(config_file, "r", encoding="utf-8") as f:
                        laya_config_dict = json.load(f)
                except Exception:
                    pass
            if laya_config_dict:
                config = _laya_config_from_dict(laya_config_dict, name_or_path=str(model_id))
            else:
                config = PretrainedConfig()
                config.model_type = "laya"
                config.name_or_path = str(model_id)
        return super().from_pretrained(
            model_id,
            config=config,
            export=export,
            force_download=force_download,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVQuantizationConfigBase, Dict] = None,
        **kwargs,
    ):
        model_path = Path(model_id)
        is_local_dir = model_path.is_dir()

        def _get_file(filename: str) -> Optional[Path]:
            if is_local_dir:
                file_path = model_path / filename
                return file_path if file_path.exists() else None
            try:
                return Path(
                    hf_hub_download(
                        repo_id=str(model_id),
                        filename=filename,
                        cache_dir=cache_dir,
                        token=token,
                        revision=revision,
                        local_files_only=local_files_only,
                        force_download=force_download,
                    )
                )
            except Exception:
                return None

        # The Laya tokenizer lives in the `tokenizer/` subfolder of the checkpoint.
        tokenizer_dir = model_path / "tokenizer" if is_local_dir else None
        if tokenizer_dir is None or not tokenizer_dir.exists():
            tokenizer_file = _get_file("tokenizer/tokenizer_config.json")
            tokenizer_dir = tokenizer_file.parent if tokenizer_file is not None else model_id
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        # rl_agent_config.json carries the inference-time parameters (temperatures, sequence budgets).
        config_file = _get_file(LAYA_CONFIG_NAME)
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                laya_config_dict = json.load(f)
        else:
            laya_config_dict = {}
        if config is None or getattr(config, "model_type", None) != "laya":
            config = _laya_config_from_dict(laya_config_dict, name_or_path=str(model_id))

        model = super()._from_pretrained(
            model_id=model_id,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            from_onnx=from_onnx,
            local_files_only=local_files_only,
            tokenizer=tokenizer,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )
        model.tokenizer = tokenizer
        model.laya_config = laya_config_dict or {
            key: getattr(model.config, key)
            for key in ("max_len", "head_max_len", "temperature", "temperature_by_options")
            if hasattr(model.config, key)
        }
        model.temperature = laya_config_dict.get("temperature") or getattr(model.config, "temperature", [1.0, 1.0, 1.0])
        model.temperature_by_options = laya_config_dict.get("temperature_by_options") or getattr(
            model.config, "temperature_by_options", {}
        )
        return model

    def _save_pretrained(self, save_directory: Union[str, Path]):
        super()._save_pretrained(save_directory)
        # Persist the assets the runtime needs to rebuild the official system_one API: the
        # tokenizer (kept in its original `tokenizer/` subfolder) and the inference parameters.
        save_directory = Path(save_directory)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.laya_config:
            with open(save_directory / LAYA_CONFIG_NAME, "w", encoding="utf-8") as f:
                json.dump(self.laya_config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        return {
            "t": t,
            "ins": qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"]),
            "crit": crit,
        }

    def system_one(self, state, questions: Dict[str, Dict]):
        """Answer a batch of typed questions about `state`, mirroring the official Laya API.

        questions: `{id: {"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}}`.
        Returns `{"model": "laya", "answers": {...}, "usage": {...}}`.
        """
        import time

        self.compile()

        cfg = self.laya_config if self.laya_config else {}
        max_len = cfg.get("max_len", 8192)
        head_max_len = cfg.get("head_max_len", 384)

        ids, items = list(questions.keys()), []
        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers = _build_sequence(self.tokenizer, state, q, max_len, head_max_len)
            if len(markers) != len(_render_options(q)):
                raise ValueError(
                    "question %r: options do not fit in head_max_len=%d tokens" % (qid, head_max_len)
                )
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = _collate_items(items, self.tokenizer.pad_token_id)

        logits, act_logits = self._infer_batch(b)
        # the official API softmaxes the act head logits before reporting act probabilities
        act_z = act_logits - act_logits.max(axis=-1, keepdims=True)
        act_p = np.exp(act_z)
        act_p = act_p / act_p.sum(axis=-1, keepdims=True)
        n_tokens = int(b["attention_mask"].sum())

        answers = {}
        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            z = logits[r, :k] / self.temperature_by_options.get(_temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": float(act_p[r, 0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(_confidence_from_probs(p, k), 4),
                    "rl_agent": ext,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(_confidence_from_probs(p, k), 4),
                    "rl_agent": ext,
                }
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
        return {"model": "rl-agent", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}}

    def _infer_batch(self, batch: Dict[str, torch.Tensor]):
        inputs = {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
            "marker_pos": batch["marker_pos"].numpy(),
            "marker_mask": batch["marker_mask"].numpy(),
            "qtype": batch["qtype"].numpy(),
        }
        # OpenVINO does not accept bool inputs through some runtimes; cast to the exported dtype.
        if self.input_dtypes.get("marker_mask") in ("i32", "i64"):
            inputs["marker_mask"] = inputs["marker_mask"].astype(np.int64)
        outputs = self._inference(inputs)
        logits = outputs["logits"]
        act_logits = outputs["act_logits"]
        return np.asarray(logits), np.asarray(act_logits)

    def forward(self, **kwargs):
        self.compile()
        np_inputs = isinstance(next(iter(kwargs.values())), np.ndarray)
        inputs = {}
        for input_name in self.input_names:
            value = kwargs.pop(input_name)
            inputs[input_name] = value if np_inputs else value.cpu().numpy()
        outputs = self._inference(inputs)
        model_outputs = {}
        for key, value in outputs.items():
            key_name = next(iter(key.names))
            if "." in key_name:
                key_name = key_name.split(".")[0]
                if key_name not in model_outputs:
                    model_outputs[key_name] = []
                model_outputs[key_name].append(value if np_inputs else torch.from_numpy(value).to(self.device))
            else:
                model_outputs[key_name] = value if np_inputs else torch.from_numpy(value).to(self.device)
        return model_outputs
