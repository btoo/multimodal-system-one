"""Build a pinned, grouped multimodal backbone-selection dataset.

Public source media lives under ignored data/. Labels are never passed to model
adapters. The fixed screen grid is independent of annotation geometry.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
import soundfile as sf

from mmso.artifacts import ROOT, sha256
from mmso.data import download, SCREEN_REVISION

PROTOCOL = ROOT / "evals/v4-selection-protocol-v1.json"
SPLITS = ("train", "calibration", "development", "confirmation")
INTENTS = {
    "alarm_set": "Set an alarm", "alarm_remove": "Remove an alarm",
    "calendar_set": "Add a calendar event", "calendar_query": "Ask about calendar events",
    "email_sendemail": "Send an email", "weather_query": "Ask about the weather",
    "calendar_remove": "Remove a calendar event", "email_query": "Ask about received emails",
}
REGIONS = [f"{row} {col}" for row in ("top", "middle", "bottom") for col in ("left", "center", "right")]


def rank(value):
    return hashlib.sha256(("miso-v4-selection-v1:" + str(value)).encode()).hexdigest()


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def media(path, modality):
    return {"path": str(path.relative_to(ROOT)), "modality": modality, "sha256": sha256(path)}


def case(identifier, split, track, group, question, choices, answer, media_items=(), **extras):
    options = list(choices)
    random.Random(rank(identifier)).shuffle(options)
    assert answer in options and len(set(options)) == len(options)
    return {"id": identifier, "split": split, "track": track, "group_id": group,
            "question": question, "choices": options, "target": options.index(answer),
            "media": list(media_items), **extras}


def text_cases(protocol):
    out = []
    for split in SPLITS:
        for i in range(protocol["tracks"]["text_rules"][split]):
            rng = random.Random(rank(f"text:{split}:{i}"))
            family = i % 4
            if family == 0:
                paid, disputed, overdue = (rng.choice([False, True]) for _ in range(3))
                answer = "Close" if paid else "Review dispute" if disputed else "Send reminder" if overdue else "Wait"
                question = ("Apply this invoice policy: close paid invoices; otherwise review a dispute before doing anything else; "
                            "otherwise remind only when overdue; otherwise wait. "
                            f"Record: paid={paid}; disputed={disputed}; overdue={overdue}. What action follows?")
                options = ["Close", "Review dispute", "Send reminder", "Wait"]
            elif family == 1:
                external, privileged, changed = (rng.choice([False, True]) for _ in range(3))
                answer = "Escalate" if external and privileged else "Review" if changed else "Record only"
                question = ("Security policy: escalate only if access is both external and privileged. "
                            "If that rule does not apply, review a changed credential; otherwise record only. "
                            f"External access: {external}. Privileged: {privileged}. Credential changed: {changed}.")
                options = ["Escalate", "Review", "Record only"]
            elif family == 2:
                requested, approved, sent = (rng.choice([False, True]) for _ in range(3))
                answer = "Already done" if sent else "Execute" if requested and approved else "Ask for approval" if requested else "No action"
                question = ("For an agent action, report already done if sent. Otherwise execute only when requested and approved. "
                            "If requested but not approved ask for approval. Otherwise take no action. "
                            f"State: requested={requested}, approved={approved}, sent={sent}.")
                options = ["Already done", "Execute", "Ask for approval", "No action"]
            else:
                severity, blocked, customer = rng.randrange(1, 5), rng.choice([False, True]), rng.choice([False, True])
                answer = "Urgent" if severity >= 3 and blocked else "Priority" if customer else "Normal"
                question = ("Support routing: urgent requires severity at least 3 AND blocked work. "
                            "Otherwise a customer-facing issue gets priority. All others are normal. "
                            f"Severity={severity}; work blocked={blocked}; customer-facing={customer}.")
                options = ["Urgent", "Priority", "Normal"]
            # Distinct case records and group identities. Policy templates intentionally
            # repeat across splits: these are learnability controls, not novel workflows.
            nonce = rng.randrange(100000, 999999)
            question += f" The unrelated ticket identifier is T{nonce}."
            out.append(case(f"rules:{split}:{i}", split, "text_rules", f"rules:{split}:{i}", question, options, answer,
                            policy_family=family, source="project-authored executable rule fixture"))
    return out


def speech_cases(protocol):
    out, used_ids, used_audio = [], set(), set()
    files = list((ROOT / "data/v4/slurp/data").glob("*.parquet"))
    sources = {}
    for name in ("train", "devel", "test"):
        path = next(p for p in files if p.name.startswith(name + "-"))
        table = pq.read_table(path)
        names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        by_label = defaultdict(dict)
        for record in table.to_pylist():
            label = names[record["intent"]]
            if label in INTENTS:
                # One human recording per semantic utterance ID; no transcript as input.
                by_label[label].setdefault(record["slurp_id"], record)
        targets = ("train",) if name == "train" else ("calibration", "development") if name == "devel" else ("confirmation",)
        for split in targets:
            quota = protocol["tracks"]["speech_intent"][split + "_per_class"]
            for label in INTENTS:
                pool = sorted(by_label[label].values(), key=lambda r: rank(r["slurp_id"]))
                selected = 0
                for record in pool:
                    identity = str(record["slurp_id"])
                    if identity in used_ids:
                        continue
                    payload = record["audio"]["bytes"]
                    fingerprint = hashlib.sha256(payload).hexdigest()
                    if fingerprint in used_audio:
                        continue
                    waveform, sr = sf.read(io.BytesIO(payload), dtype="float32")
                    if len(waveform) / sr > 30:
                        continue
                    used_ids.add(identity); used_audio.add(fingerprint)
                    path_out = ROOT / "data/v4/media/slurp" / f"{identity}.wav"
                    path_out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path_out, waveform, sr, subtype="PCM_16")
                    out.append(case("slurp:" + identity, split, "speech_intent", "slurp:" + identity,
                                    "What is the intent of the speaker's request?", list(INTENTS.values()), INTENTS[label],
                                    [media(path_out, "audio")], source_split=name,
                                    source="SLURP real recordings via qmeeus/slurp pinned mirror", source_intent=label))
                    selected += 1
                    if selected == quota:
                        break
                if selected != quota:
                    raise ValueError(f"Insufficient unique SLURP {split}/{label}: {selected}/{quota}")
        sources[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    return out, sources


def sound_cases(protocol):
    old = read_rows(ROOT / "evals/manifests/sound_events.jsonl")
    out = []
    for split, old_split in zip(SPLITS, ("train", "calibration", "dev", "test")):
        for label in sorted({r["target"] for r in old}):
            quota = protocol["tracks"]["sound_events"][split + "_per_class"]
            pool = sorted((r for r in old if r["split"] == old_split and r["target"] == label), key=lambda r: rank(r["id"]))
            assert len(pool) >= quota
            for row in pool[:quota]:
                options = [x.replace("_", " ") for x in row["labels"]]
                out.append(case(row["id"], split, "sound_events", row["group_id"],
                                "Which sound is most clearly audible in this recording?", options, label.replace("_", " "),
                                row["media"], source_fold=row["source_fold"], source="ESC-10 public recordings"))
    return out


def screen_cases(protocol):
    out, used_images = [], set()
    settings = protocol["tracks"]["screen_region"]
    base = f"https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{SCREEN_REVISION}/"
    for app in settings["applications"]:
        annotation = ROOT / "data/screens_v2/annotations" / (app + ".json")
        records = json.loads(annotation.read_text())
        for split in SPLITS:
            for kind in ("text", "icon"):
                quota = settings["per_application_per_type"][split]
                pool = sorted((r for r in records if r["ui_type"] == kind), key=lambda r: rank(r["id"]))
                selected = 0
                for record in pool:
                    name = record["img_filename"]
                    if name in used_images:
                        continue
                    used_images.add(name)
                    path = ROOT / "data/screens_v2/images" / name
                    download(base + "images/" + quote(name, safe="/"), path, max_bytes=25 << 20)
                    width, height = record["img_size"]
                    x0, y0, x1, y1 = record["bbox"]
                    col = min(2, int((x0 + x1) / 2 / width * 3))
                    row = min(2, int((y0 + y1) / 2 / height * 3))
                    question = ("Divide the entire screenshot into a fixed 3-by-3 grid of equal rectangles. "
                                "Which region contains the center of the UI element requested below? "
                                f"Requested element: {record['instruction']}")
                    out.append(case("screen:" + record["id"], split, "screen_region", "screen:" + name,
                                    question, REGIONS, REGIONS[row * 3 + col], [media(path, "image")],
                                    application=app, ui_type=kind, source="ScreenSpot-Pro fixed-grid derived task"))
                    selected += 1
                    if selected == quota:
                        break
                if selected != quota:
                    raise ValueError(f"Insufficient unique screens {app}/{kind}/{split}")
        print(f"Prepared {app} screenshots", flush=True)
    return out


def joint_cases(protocol):
    old = read_rows(ROOT / "evals/manifests/speech_keywords.jsonl")
    out, used_panels = [], set()
    words = sorted({r["target"] for r in old})
    colors = {"red": "#e96a61", "blue": "#75a6ee", "green": "#81c58b", "yellow": "#f0d871"}
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36)
    for split, old_split in zip(SPLITS, ("train", "calibration", "dev", "test")):
        pool = sorted((r for r in old if r["split"] == old_split), key=lambda r: rank(r["id"]))
        count = protocol["tracks"]["joint_control"][split]
        for i, row in enumerate(pool[:count]):
            rng = random.Random(rank(row["id"]))
            while True:
                selected_words = rng.sample(words, 4)
                color_names = list(colors)
                rng.shuffle(color_names)
                signature = tuple(zip(selected_words, color_names))
                if signature not in used_panels:
                    used_panels.add(signature)
                    break
            image = Image.new("RGB", (640, 400), "white")
            draw = ImageDraw.Draw(image)
            for idx, (word, color) in enumerate(zip(selected_words, color_names)):
                x, y = (idx % 2) * 320, (idx // 2) * 200
                draw.rectangle((x + 8, y + 8, x + 311, y + 191), fill=colors[color])
                draw.text((x + 160, y + 100), word.upper(), fill="#152018", font=font, anchor="mm")
            answer = color_names[selected_words.index(row["target"])] if row["target"] in selected_words else "not present"
            path = ROOT / "data/v4/media/joint" / f"{split}-{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            out.append(case(f"joint:{split}:{i}", split, "joint_control", row["group_id"],
                            "Listen to the spoken word. Which colored tile contains that exact word? Select not present if absent.",
                            [*colors, "not present"], answer, [media(path, "image"), *row["media"]],
                            source="Project-generated panels plus public human Speech Commands audio"))
    return out


def validate(rows):
    ids, groups, media_splits = set(), {}, {}
    for row in rows:
        assert row["id"] not in ids; ids.add(row["id"])
        prior = groups.setdefault(row["group_id"], row["split"])
        assert prior == row["split"], ("group overlap", row["group_id"])
        for item in row["media"]:
            assert sha256(ROOT / item["path"]) == item["sha256"]
            prior = media_splits.setdefault(item["sha256"], row["split"])
            assert prior == row["split"], ("media overlap", item["path"])
    return {"examples": len(rows), "groups": len(groups), "unique_media": len(media_splits),
            "counts": {s: dict(Counter(r["track"] for r in rows if r["split"] == s)) for s in SPLITS},
            "group_and_exact_media_disjoint": True}


def main():
    destination = ROOT / "evals/manifests/v4_selection_v1.jsonl"
    if destination.exists():
        print(json.dumps(validate(read_rows(destination)), indent=2)); return
    protocol = json.loads(PROTOCOL.read_text())
    speech, sources = speech_cases(protocol)
    rows = text_cases(protocol) + speech + sound_cases(protocol) + screen_cases(protocol) + joint_cases(protocol)
    audit = validate(rows)
    destination.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    acquisition = {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(destination),
                   "audit": audit, "slurp": {"repo": "qmeeus/slurp", "revision": "91b0abfee2e735282967ee00d631d6d5f0fb7ff9",
                   "source_shards": sources, "original": "https://github.com/pswietojanski/slurp", "policy": "Use real-recording shards only; one recording per utterance ID; exclude IDs seen in another partition; never pass transcripts to the model"},
                   "screenspot_revision": SCREEN_REVISION, "official_benchmark_result": False,
                   "upstream_pretraining_contamination": "unknown", "labels_in_model_inputs": False}
    (ROOT / "evals/acquisition/v4_selection_v1.json").write_text(json.dumps(acquisition, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
