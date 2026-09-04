# Handoff: RT-DETR training on the RTX 3060

You are running one arm of a four-model detector comparison. Another machine
(RTX 5060) is running the other three. **Your job is to train RT-DETR-l for 50
epochs on the BDD100K daytime+clear subset and send back three files.**

Do not evaluate, benchmark, or analyse anything. Train and return artifacts.
Everything else happens on the other machine, for reasons explained below.

---

## 1. Non-negotiables

The value of this project is that four models are compared under identical
conditions. Breaking any of these silently invalidates the comparison — the run
will complete and produce plausible numbers that mean nothing.

| Must not change | Why |
|---|---|
| The image subset | All four models must train on byte-identical data. Verify with §4 before training. |
| `epochs=50` | Matched training budget is the control. See the note in §6 before you decide RT-DETR "needs more". |
| `imgsz=640` | Input resolution is matched across all four models. |
| Class order in `src/bddcv/constants.py` | It is the contract binding labels, annotations and every results table. |
| `seed=0` | Reproducibility. |

**Do not measure inference speed or FPS on this machine.** Accuracy transfers
across GPUs; throughput does not. All four models get benchmarked on one card
at the end. A speed number from the 3060 would confound hardware with
architecture.

---

## 2. What to copy from the 5060 machine

Copy these into a fresh project directory. **Do not copy `archive.zip`,
`data/bdd100k*/`, or `runs/`** — they are large and unnecessary.

```
scripts/                       # all of it
src/                           # all of it
configs/bdd_source.yaml
constraints.txt
data/source_daytime_clear/     # ~1.1 GB - images, labels, annotations, manifests
```

Total ≈ 1.1 GB.

---

## 3. Environment

Python 3.14 is what the other machine runs; 3.11+ is fine. Do the GPU check
first — if it fails, nothing else matters.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0)); torch.zeros(8,device='cuda').sum().item()"
```

Expect `(8, 6)` for the RTX 3060 (Ampere). The final statement is the real
test — capability reporting can succeed while kernel launches still fail.

If torch is missing or CPU-only:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics pycocotools -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**The `-c constraints.txt` is not optional.** A plain `pip install ultralytics`
tries to upgrade torch from PyPI, which on Windows is the CPU-only wheel. It
does not error — it silently disables GPU training, and you discover it hours
later from the epoch times.

### Fix the dataset path

`configs/bdd_source.yaml` contains an absolute path from the other machine:

```yaml
path: D:/CV/data/source_daytime_clear
```

Change it to your absolute path. Leave `train:`, `val:` and the `names:` block
exactly as they are.

---

## 4. Verify the data before training

Do not skip this. A subset that differs by even one image makes your result
non-comparable, and you will not find out until the results are being written up.

```bash
python -c "
import hashlib, json
from pathlib import Path
d = Path('data/source_daytime_clear')
for f in ['train_images.txt','val_images.txt']:
    b = (d/f).read_bytes()
    print(f, hashlib.sha256(b).hexdigest()[:16], len(b.splitlines()))
for f in ['annotations/instances_train.json','annotations/instances_val.json']:
    j = json.loads((d/f).read_text())
    print(f, len(j['images']), len(j['annotations']))
print('train jpgs', len(list((d/'images/train').glob('*.jpg'))))
print('val   jpgs', len(list((d/'images/val').glob('*.jpg'))))
"
```

Expected output — every value must match exactly:

```
train_images.txt  90142428399da90d  12454
val_images.txt    70c991ed512100e7  1764
instances_train.json  12454  239901
instances_val.json     1764   34096
train jpgs 12454
val   jpgs 1764
```

If anything differs, stop and report it. Do not attempt to regenerate the
subset locally — re-running the pipeline against a different archive copy is
exactly how the two machines end up training on different data.

---

## 5. Smoke test first

RT-DETR-l is heavier than YOLO. Confirm it fits in 12 GB before committing to a
long run.

```bash
python -c "
from ultralytics import RTDETR
RTDETR('rtdetr-l.pt').train(data='configs/bdd_source.yaml', epochs=1, imgsz=640,
    batch=8, amp=True, device=0, workers=4, project='runs', name='smoke_rtdetr', exist_ok=True)
"
```

This downloads `rtdetr-l.pt` automatically (needs internet) and takes roughly
15–25 minutes.

- **If it OOMs**, retry with `batch=4`. Reduce batch, never `imgsz` — BDD100K is
  dense with small objects (traffic lights, signs) and lowering resolution
  changes what the comparison measures.
- **Record the per-epoch time and peak VRAM.** Report them back; they set the
  ETA and they go in the writeup.

Then delete `runs/detect/runs/smoke_rtdetr/` so it does not get confused with
the real run.

---

## 6. The real run

Launch it **detached**, not as a child of your agent session. On the other
machine, a supervisor running as a session child was killed along with the
session, which defeated its entire purpose.

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force runs\logs | Out-Null
Start-Process -FilePath "python" `
  -ArgumentList "-u","scripts/train_with_resume.py","ultra","--model","rtdetr-l.pt","--name","rtdetr-l","--epochs","50","--batch","8" `
  -WorkingDirectory "$PWD" `
  -RedirectStandardOutput "runs\logs\rtdetr.log" `
  -RedirectStandardError  "runs\logs\rtdetr.err" `
  -WindowStyle Hidden -PassThru
```

**Linux:**

```bash
mkdir -p runs/logs
nohup python -u scripts/train_with_resume.py ultra --model rtdetr-l.pt \
  --name rtdetr-l --epochs 50 --batch 8 > runs/logs/rtdetr.log 2>&1 &
```

`train_with_resume.py` supervises the run: if training dies, it relaunches with
resume from the last checkpoint. It stops itself if three consecutive restarts
complete no epoch, which means a real failure rather than a transient fault.
Use `--batch 4` here too if the smoke test needed it.

> **On epochs.** RT-DETR is known to converge more slowly than YOLO and is
> often trained for 72+ epochs. Do not raise it. Every model in this comparison
> gets the same 50-epoch budget; that is the control. RT-DETR's score is
> reported as a lower bound under a matched budget, and that limitation is
> disclosed in the writeup. Changing it here would quietly break the one thing
> the project is measuring.

---

## 7. Monitoring

```bash
grep -E "^\[wrapper|ABORT|COMPLETE" runs/logs/rtdetr.log   # restarts and outcome
tail -3 runs/detect/runs/rtdetr-l/results.csv               # per-epoch metrics
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
```

Success looks like a final log line reading `COMPLETE: 50 epochs in <N>h`.
Failure looks like `ABORT:` followed by a reason — send the log if you see one.

---

## 8. Send back

Three files:

| File | Path |
|---|---|
| Best checkpoint | `runs/detect/runs/rtdetr-l/weights/best.pt` |
| Per-epoch metrics | `runs/detect/runs/rtdetr-l/results.csv` |
| Supervisor log | `runs/logs/rtdetr.log` |

Plus a short note with: peak VRAM, per-epoch time, total wall-clock, and how
many restarts the wrapper performed.

`best.pt` is the checkpoint at the highest validation mAP@[.5:.95], which is
**not necessarily the last epoch** — on the 5060's YOLO run the best was epoch
39 of 50. Send `best.pt`, not `last.pt`.

Do not run `scripts/evaluate.py` here. Reported numbers come from a single
`pycocotools` path executed on one machine, so that no model is scored by a
different implementation than another. Ultralytics' own mAP is not COCO mAP and
must not be quoted.

---

## 9. If something goes wrong

| Symptom | Cause | Action |
|---|---|---|
| `no kernel image is available` | torch built without your GPU's arch | Reinstall from the cu128 index (§3) |
| Epochs take absurdly long, GPU idle | CPU-only torch got installed | Reinstall with `-c constraints.txt` |
| CUDA OOM at start | batch too large for 12 GB | Drop to `--batch 4`, never lower `imgsz` |
| Dataset not found | `path:` in the yaml still points at `D:/CV` | Fix it (§3) |
| Fingerprints in §4 don't match | Incomplete or wrong data copy | Stop, re-copy, report |
| `ABORT: consecutive restarts completed no epoch` | Real failure, not a driver fault | Send `runs/logs/rtdetr.err` |

Anything not on this list, or any doubt about whether a deviation is
acceptable: **ask before proceeding.** A run that completes with the wrong
settings costs more than a run that pauses for a question.
