# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "torch", "requests"]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        """
        # KittyBrain 300M
        Trains a ~300M-parameter char-level GPT for the mcs desktop pet:
        **TinyStories** (simple English, for language structure) mixed with a
        mood-tagged **cat corpus** (for personality and swearing).

        Time-boxed to fit molab's free GPU window: it trains for
        `TRAIN_MINUTES`, checkpointing every 10 minutes, then offers
        `kitty_brain.pt` for download. Drop that file into
        `~/desktop-pet/brain/` on the laptop (replacing the 22M one) and the
        pet uses it automatically.

        **Run cells top to bottom. Make sure the runtime has a GPU.**
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import json
    import math
    import os
    import random
    import time

    import requests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = (torch.cuda.get_device_name(0) if DEVICE == "cuda"
                else "NO GPU — enable a GPU runtime or this will take days")
    TRAIN_MINUTES = 70          # leave margin inside the 90-minute window
    CTX, D, LAYERS, HEADS, FF = 256, 1024, 24, 16, 4096
    torch.manual_seed(7)
    random.seed(7)
    mo.md(f"**Device:** `{DEVICE}` — {GPU_NAME}")
    return (CTX, D, DEVICE, F, FF, HEADS, LAYERS, TRAIN_MINUTES, json, math,
            nn, os, random, requests, time, torch)


@app.cell
def _(mo, os, random, requests):
    # ---- cat corpus (mood-tagged, spicy) + TinyStories grounding ----------
    THINGS = ["fish", "the box", "yarn", "the dot", "birbs", "treats",
              "naps", "the obby", "my turtle", "the kitten", "snacks",
              "toes", "the laser", "sunbeams", "crumbs", "the wall",
              "my tail", "bugs", "the cursor", "dinner", "shrimp", "tuna"]
    VERBS = ["want", "need", "love", "demand", "require", "chase",
             "protec", "judge", "bap", "sniff", "hunt", "dream of"]
    FEELS = ["happy", "cozy", "sleepy", "mighty", "smol", "round", "fast",
             "fluffy", "important", "majestic", "peckish"]
    NOISES = ["mrrp", "mrow", "prrb", "meep", "nya", "mew", "purr"]
    EXCL = ["!!", "!", "", " :3", " hehe", " fr", ""]

    def _r(seq):
        return random.choice(seq)

    MOODS = {
        "happy": [
            lambda: "i am %s%s" % (_r(FEELS), _r(EXCL)),
            lambda: "%s%s" % (_r(NOISES), _r(EXCL)),
            lambda: "i %s %s%s" % (_r(VERBS), _r(THINGS), _r(EXCL)),
            lambda: "fuck yeah %s" % _r(THINGS),
            lambda: "im the shit", lambda: "damn im cute",
            lambda: "best fucking day", lambda: "im him. the cat",
            lambda: "purring at max power",
        ],
        "ok": [
            lambda: "*%s quietly*" % _r(["stares", "loafs", "vibes",
                                         "blinks", "sits"]),
            lambda: "thinking about %s" % _r(THINGS),
            lambda: "just vibing n shit", lambda: "bored as hell",
            lambda: "*slow blink*", lambda: "meh af",
            lambda: "cats dont give a shit",
        ],
        "hungry": [
            lambda: "i %s %s" % (_r(["want", "need", "demand", "require"]),
                                 _r(["food", "snacks", "treats", "fish",
                                     "dinner", "tuna", "shrimp"])),
            lambda: "where the fuck is dinner", lambda: "feed me damn it",
            lambda: "bowl status: empty", lambda: "hangry as fuck",
            lambda: "goddamn empty bowl", lambda: "*sad hungry meow*",
            lambda: "feed me you bastard",
        ],
        "mad": [
            lambda: "no.", lambda: "unacceptable.",
            lambda: "this is bullshit", lambda: "son of a bitch",
            lambda: "fuck this %s" % _r(["obby", "wall", "day", "vacuum"]),
            lambda: "oh for fucks sake", lambda: "piss off",
            lambda: "pure fucking rage", lambda: "i bite soon",
            lambda: "im telling the kitten",
        ],
        "worry": [
            lambda: "battery low!! charge!!", lambda: "oh shit the battery",
            lambda: "the pc is fucked??", lambda: "oh crap oh crap",
            lambda: "were screwed??",
        ],
        "vamp": [
            lambda: "bleh bleh bleh", lambda: "i vant %s" % _r(THINGS),
            lambda: "ze fucking night", lambda: "creature of ze night",
            lambda: "ze fridge calls to me", lambda: "fear me damn it",
        ],
        "dream": [
            lambda: "*dreams of %s*" % _r(THINGS),
            lambda: "zzz %s zzz" % _r(NOISES), lambda: "*paw twitch*",
            lambda: "zzz fuck zzz", lambda: "*swears in sleep*",
            lambda: "5 more naps",
        ],
    }

    cat_lines = []
    for mood, temps in MOODS.items():
        for _i in range(6000):
            cat_lines.append("<%s>%s" % (mood, _r(temps)()[:26]))
    random.shuffle(cat_lines)
    cat_text = "\n".join(cat_lines) + "\n"

    ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?*:<>-\n")
    TS_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
              "resolve/main/TinyStories-train.txt")
    TS_CAP = 120_000_000        # chars of TinyStories to keep

    ts_path = "tinystories.txt"
    if not os.path.exists(ts_path):
        with requests.get(TS_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            got = 0
            with open(ts_path, "w") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    s = chunk.decode("utf-8", "ignore").lower()
                    s = "".join(c for c in s if c in ALLOWED)
                    f.write(s)
                    got += len(s)
                    if got >= TS_CAP:
                        break
    ts_text = open(ts_path).read()[:TS_CAP]

    # cat corpus ~20% of the stream so the persona stays dominant per epoch
    reps = max(1, int(len(ts_text) * 0.25 / max(1, len(cat_text))))
    text = ts_text + cat_text * reps
    chars = sorted(set(text))
    mo.md("corpus: **%.1fM chars** (%.1fM stories + %d× cat corpus), "
          "vocab %d" % (len(text) / 1e6, len(ts_text) / 1e6, reps,
                        len(chars)))
    return chars, text


@app.cell
def _(CTX, D, DEVICE, FF, HEADS, LAYERS, chars, mo, nn, text, torch):
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.uint8)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
            self.ln2 = nn.LayerNorm(D)
            self.mlp = nn.Sequential(nn.Linear(D, FF), nn.GELU(),
                                     nn.Linear(FF, D))

        def forward(self, x, mask):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class KittyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(V, D)
            self.pos = nn.Parameter(torch.zeros(1, CTX, D))
            self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
            self.lnf = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)

        def forward(self, idx):
            T = idx.shape[1]
            mask = torch.triu(torch.full((T, T), float("-inf"),
                                         device=idx.device), diagonal=1)
            x = self.tok(idx) + self.pos[:, :T]
            for b in self.blocks:
                x = b(x, mask)
            return self.head(self.lnf(x))

    model = KittyGPT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    mo.md("model: **%.1fM parameters** on %s" % (n_params / 1e6, DEVICE))
    return KittyGPT, V, data, model, stoi


@app.cell
def _(CTX, D, DEVICE, F, FF, HEADS, LAYERS, TRAIN_MINUTES, V, chars, data,
      math, mo, model, time, torch):
    # ---- time-boxed training ----------------------------------------------
    BATCH = 32 if DEVICE == "cuda" else 4
    LR = 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    use_bf16 = (DEVICE == "cuda"
                and torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"
                                           and not use_bf16))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    def get_batch():
        ix = torch.randint(len(data) - CTX - 1, (BATCH,))
        x = torch.stack([data[i:i + CTX] for i in ix]).long().to(DEVICE)
        y = torch.stack([data[i + 1:i + CTX + 1]
                         for i in ix]).long().to(DEVICE)
        return x, y

    def save_ckpt(step):
        state = {k: v.half().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": state, "chars": chars,
                    "cfg": {"ctx": CTX, "d": D, "layers": LAYERS,
                            "heads": HEADS, "ff": FF, "vocab": V},
                    "step": step}, "kitty_brain.pt")

    deadline = time.time() + TRAIN_MINUTES * 60
    next_save = time.time() + 600
    step = 0
    log = []
    est_total = TRAIN_MINUTES * 60 / 0.25   # rough, refined as it runs
    while time.time() < deadline:
        step += 1
        frac = 1 - (deadline - time.time()) / (TRAIN_MINUTES * 60)
        lr = LR * min(1.0, step / 300) * (0.55 + 0.45 * math.cos(
            math.pi * min(1.0, frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch()
        with torch.autocast(device_type=DEVICE, dtype=amp_dtype,
                            enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % 50 == 0:
            msg = "step %5d  loss %.3f  %4.1f min left" % (
                step, loss.item(), (deadline - time.time()) / 60)
            log.append(msg)
            print(msg, flush=True)
        if time.time() >= next_save:
            save_ckpt(step)
            next_save += 600
            print("checkpoint saved at step", step, flush=True)
    save_ckpt(step)
    final_loss = loss.item()
    mo.md("**training done** — %d steps, final loss %.3f\n\n```\n%s\n```"
          % (step, final_loss, "\n".join(log[-12:])))
    return (final_loss,)


@app.cell
def _(CTX, DEVICE, chars, final_loss, mo, model, stoi, torch):
    # ---- sample the finished brain ----------------------------------------
    _ = final_loss
    MOOD_LIST = ["happy", "ok", "hungry", "mad", "worry", "vamp", "dream"]
    model.eval()
    rows = []
    with torch.no_grad():
        for mood in MOOD_LIST:
            outs = []
            for _n in range(4):
                idx = [stoi.get(c, 0) for c in "\n<%s>" % mood]
                s = ""
                for _t in range(30):
                    x = torch.tensor([idx[-CTX:]], device=DEVICE)
                    logits = model(x)[0, -1] / 0.9
                    v, ix = torch.topk(logits, 40)
                    p = torch.softmax(v, dim=-1)
                    ch = chars[ix[torch.multinomial(p, 1)].item()]
                    if ch == "\n":
                        break
                    s += ch
                    idx.append(stoi[ch])
                outs.append(s.strip())
            rows.append("**%s** — %s" % (mood, " · ".join(outs)))
    mo.md("\n\n".join(rows))
    return


@app.cell
def _(mo):
    import pathlib
    p = pathlib.Path("kitty_brain.pt")
    if p.exists():
        dl = mo.download(data=p.read_bytes(), filename="kitty_brain.pt",
                         label="download kitty_brain.pt (~600MB)")
    else:
        dl = mo.md("no checkpoint yet — run the training cell first")
    dl
    return


if __name__ == "__main__":
    app.run()
