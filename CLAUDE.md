# Rules for this project

## Start here (state as of 2026-10-17, early afternoon)

- **Brief:** `case.md` (in Russian). The deadline is 12:00 on 2026-10-18.
- **Approved design:** `design.md`.
- **Implementation plan:** `docs/superpowers/plans/2026-10-17-cctv-review-service.md`, 11 tasks, each with an owner and its dependencies. No task has started yet, and `svc/` doesn't exist yet.
- **Training data:** `open/videos/` (42 .mov files) and `open/open_labels.xlsx`.
- **Open decisions.** Ask the user; never assume these:
  1. Approval for the pod setup in Task 6: create `/home/jovyan/model_train_amir/`, build three environments, download about 45 GB of weights and upload the clips. It changes the server, so it needs an explicit yes.
  2. How to run the plan: one session doing every task itself, or a fresh subagent per task.
  3. How the public URL reaches the pod. This blocks Task 8.
- **Reaching the pod:** use `~/Coding/armeta/ge-2052-rpn-classifier/tools/jx.py`, which runs shell commands on JupyterHub, and `jup_upload.py`, which uploads files. Both need a local venv with `requests` and `websocket-client`; create one in the session's scratchpad. The token is in Keychain. Never print it.

## Project name

The project is called **Kuá** (Kazakh "куә", "witness"). The user chose the name on
2026-10-17. Use it in the pitch deck, the UI title, the README and anywhere else the
product is named. The idea behind it: the service shows what it saw and does not accuse
anyone, which matches the brief's line that the result is a hint for a human reviewer.
The name contains none of the banned words below, so it may appear on the server, but
the server root stays `/home/jovyan/model_train_amir/`.

## Naming on the server (Armeta JupyterHub, jupyter.armeta.kz, user `amirizimov`)

Nothing on the server may carry the hackathon's or the client's name. The words to avoid
are `forte`, `fortebank`, `datathon` and `hackathon`, in any case and in any language
(for example `форте`, `датaтон`, `хакатон`).

- **Everything for this project lives under `/home/jovyan/model_train_amir/`.** That
  covers code, venvs, model weights, uploaded videos, job files, logs, caches and outputs.
  The pod's `/` is read-only and not persistent, so `/model_train_amir/` at the root is
  not an option. `/home/jovyan` is the persistent volume.
- **Processes follow the same rule.** Launch every related process from scripts inside
  `/home/jovyan/model_train_amir/`, with neutral script, argument and log names, so
  `ps`, `nvidia-smi` and the logs show only `model_train_amir` paths. This includes the
  API server, model workers, any tunnel, Jupyter kernels and terminals, MLflow
  experiment and run names, env var names, and service or tunnel hostnames.
- **Do not write anything outside that folder** for this project. If a tool insists on
  a default location (for example the Hugging Face cache), point it inside the folder
  (`HF_HOME=/home/jovyan/model_train_amir/hf`).
- **The rule covers names.** If file contents shown to outside users (a UI title, for
  example) would need the client's name, ask first.

## GPU sharing on that pod

There is one H200, shared with the GE-2052 "infinite" training runs, which are relaunched
by `~/ge-2052/code/rv20261007_keepalive.sh`. Its stop switch is the file
`~/ge-2052/logs/rd2/KEEPALIVE.STOP`. Never stop, kill or pause those runs, and never
create or remove the stop file, without the user's approval for that specific action.

The one approval given so far (2026-10-17) applies only at the moment our models are
first started on the GPU, not before: run `touch ~/ge-2052/logs/rd2/KEEPALIVE.STOP`,
then `kill -TERM` the yolo1280_s0 trainer. Find the trainer's PID again at that moment,
because a new generation gets a new PID; it is the `rv20261006_train_yolo.py` process
whose arguments include `--hyp infinite-yolo1280_s0`. Confirm the freed memory with
`nvidia-smi` afterwards. To undo, remove the stop file and relaunch the keepalive
pointed at each run's newest `last.pt`.
