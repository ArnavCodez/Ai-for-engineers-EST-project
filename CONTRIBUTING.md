# Contributing: team rules for the EST project

Every teammate follows these rules in every work session. `LOG.md`, `CONCEPTS.md`, `VIVA_QA.md` and `PLAN.md` refer to its sections. (It is the EST copy of the rules we used for our MST project.)

---

## 1. Context

- **Course:** UCS321 AI for Engineers, TIET Patiala, B.E. Electronics & Computer Engineering, Semester III (odd sem 2026).
- **Team:** a group of 5 students. Different teammates run different sessions.
- **Project:** EST statement #3, Tata Steel: a **CNN** that recognises surface defects and cracks in steel plate images. The plan is in `PLAN.md`.
- **How we are graded:** code/report marks plus a **presentation and viva**. Finishing is not enough. **Every teammate must be able to explain every step in the viva**, which is why the logging rules in section 4 are mandatory.

---

## 2. How we work

1. **Go phase by phase.** When you take on Phase N, read that section of `PLAN.md`, do only that phase, check every acceptance criterion, then stop and write the phase summary (section 4.5). Don't jump ahead to the next phase unless the team agrees.
2. **Start of every session:**
   - Run `git pull`. If `git status` still shows your branch is behind, sort that out before doing anything else.
   - Read the last 3 entries of `LOG.md` to see where we are.
   - Read the phase checklist at the top of `PLAN.md`.
   - Before you start, be clear on what was done last and what comes next. The latest "Handover" entry in `LOG.md` (section 4.6) tells you.
   - Put your name in the "run by" field of every `LOG.md` entry you write.
3. **Never invent numbers.** Every metric, shape, count or time written in any file must come from code you actually ran. Save raw results to `outputs/results/` (CSV or JSON). Slides and reports must quote these files exactly.
4. **Use the test set once only.** It is evaluated once, in the evaluation phase, with the final chosen models. All tuning and model choice use training/validation data or cross-validation. If the test set is touched earlier by mistake, log it as `[BUG]` and tell the team.
5. **Reproducibility:**
   - Use `SEED = 42` everywhere (python `random`, numpy, scikit-learn `random_state`, TensorFlow).
   - Save the split (which profiles or images are train/val/test) to a file, so every teammate gets the same split.
6. **Paths:** the project folders sit in a path with spaces and brackets (e.g. `D:\TIET Stuff\Semester 3\AI for Engineers (UCS321)\...`).
   - In Python, always use `pathlib.Path` and paths relative to the project folder.
   - Always quote paths in shell commands.
7. **Data:**
   - Raw data goes in `data/raw/` and is never modified. For EST the notebooks read it from Kaggle (see `data/README.md`).
   - Processed data goes in `data/processed/`.
   - Never commit data, model files or `.venv/` to git. Each project has a `.gitignore` for this; don't weaken it.
8. **Keep it simple.** This is a student project: plain functions, no classes unless really needed, only the libraries listed in `PLAN.md`. Before adding anything not in the plan, ask the team, and log the decision as `[DECISION]`.
9. **Errors are learning material.** When something fails (install error, crash, wrong-looking result), fix it, then log the symptom, cause and fix as `[BUG]`.
10. **Tick progress.** When a phase meets all its acceptance criteria, tick it (`[x]`) in the checklist at the top of `PLAN.md`, and tick your tasks in the team tracker.
11. **Write down every "why".** When you or a teammate asks a "why" or "what is" question during a session and the answer isn't trivial:
    - write the answer in simple words;
    - add the concept to `CONCEPTS.md` (format in section 4.7);
    - add the question and answer to `VIVA_QA.md` (format in section 4.8).
12. **Git.** Commit and push at the end of every session (commands in `README.md`), so nobody works on an old `LOG.md`.
    - Only one person runs a phase at a time, so `LOG.md` never conflicts.
    - After a `git pull`, a merge or a `git stash pop`, run `git status` and `git diff --check` (it flags leftover conflict markers) before you commit.

---

## 3. Code style

- **Python 3.11**, with one virtual environment per project (`.venv/`, git-ignored) and a pinned `requirements.txt` per project.
- **Main work goes in Jupyter notebooks**, because they're what we present. Number them in order: `notebooks/01_data_check.ipynb`, `notebooks/02_eda.ipynb`, and so on. Small shared helper functions go in `src/*.py`.
- **Every notebook runs top to bottom** ("Restart & Run All") without errors. After editing one, execute it so the outputs are saved inside the file:
  ```
  jupyter nbconvert --to notebook --execute --inplace "notebooks/<file>.ipynb"
  ```
- **Explain in markdown cells:**
  - Each notebook starts with a markdown cell: **Goal**, **Inputs**, **Outputs**.
  - Each major code cell comes after a short markdown cell that says *what* the next cell does and *why*, in plain English.
- **Simple, beginner-readable code:**
  - standard names (`df`, `X_train`, `y_test`, `model`, `history`);
  - no clever one-liners;
  - short inline comments only, because the full explanation lives in markdown cells and `LOG.md`.
- **Every plot:**
  - has a title, axis labels with units, and a legend if it shows more than one series;
  - is saved to `outputs/figures/` as PNG (`dpi=150`);
  - has a descriptive filename, e.g. `eda_03_winding_temp_vs_current.png`.

---

## 4. LOGGING: MANDATORY, AND THE MOST IMPORTANT RULE

We are a team, and everyone must understand everything for the viva. So **log everything and every nuance as you go**, not only at the end of a phase. If in doubt, log it.

### 4.1 Three learning files (inside each project folder)

| File | What goes in it | Rules |
|---|---|---|
| `LOG.md` | A journal of everything done, in time order | **Append-only.** Never delete or rewrite old entries. Corrections go in a new entry that references the old one. |
| `CONCEPTS.md` | Every ML/DL/domain concept we used, explained simply | Grouped by topic. Improve an entry if understanding improves. |
| `VIVA_QA.md` | Every question a teacher could ask, with a short answer | Grouped by topic. Add questions as you go. |

### 4.2 When to write a LOG.md entry

Write one after **every meaningful step**, for example:
- installing packages
- downloading or checking data
- every EDA finding
- every pre-processing choice
- every model trained
- every bug
- every result

Small related steps can share one entry, but nothing important may be skipped.
**Rule of thumb:** if a teammate who wasn't in the session could be asked about it in the viva, it goes in the log.

### 4.3 LOG.md entry format (use exactly this)

```
## YYYY-MM-DD HH:MM | Phase N.x | <short title>  (run by: <name>)
**Tags:** [SETUP] [DATA] [EDA] [DECISION] [NUANCE] [BUG] [RESULT] [IDEA]   <- keep only the ones that apply
**What we did:** 1–3 lines.
**Why:** the reason, in plain English.
**How:** files/notebooks/functions touched, key parameters (e.g. `RandomForestRegressor(n_estimators=300, max_depth=12)`), commands run.
**Result:** actual numbers / shapes / timings, and figure paths, each figure with one line saying what it shows.
**Nuances / gotchas:** anything subtle: traps, surprises, warnings, things that look wrong but aren't, things that almost went wrong.
**Alternatives considered:** what else we could have done, and why we didn't.
**Viva angle:** the question this step invites + a 1–2 line answer (also copy it into VIVA_QA.md).
```

Write "—" in a field that doesn't apply. Never drop a field.

### 4.4 Tags (so teammates can Ctrl+F)

- `[DECISION]`: any choice between options (split method, feature, model, hyperparameter, metric, threshold). Always say what the options were and why we picked this one.
- `[NUANCE]`: subtle points, such as:
  - leakage risks
  - why a number looks too good or too bad
  - library quirks
  - physics or domain facts
  - "this works only because…"
- `[BUG]`: the error message (shortened), the cause and the fix.
- `[RESULT]`: any metric or finding.
- `[IDEA]`: something to try later, or a point for the "future work" slide.
- `[SETUP]`, `[DATA]`, `[EDA]`: self-explanatory.

### 4.5 End of every phase: a summary entry

Add an entry titled **"Phase N summary"** containing:
1. a 5-bullet recap in simple words;
2. the phase's acceptance checklist with ticks;
3. **"What every teammate must be able to explain from this phase"** (3–5 points);
4. the key numbers from this phase in a small table.

### 4.6 End of every session: a handover entry

Add a short entry titled **"Handover"** with:
- where we stopped;
- what's next (exact next step);
- open issues and questions;
- anything the next person must know before starting.

### 4.7 CONCEPTS.md entry format

```
### <Concept name>
- **In one line:** ...
- **Explained simply:** 3–6 lines, no jargon, an analogy if it helps.
- **Formula (if any):** ...
- **Why we used it here:** ...
- **Where in our code:** notebook + cell, or src function.
- **Common confusion / trap:** ...
```

Add a concept the **first time** it is used, e.g. GroupKFold, EWMA, RMSE, R², data leakage, regularisation, convolution, dropout, transfer learning, Grad-CAM.

### 4.8 VIVA_QA.md entry format

```
**Q:** <question as a teacher would ask it>
**A:** <1–3 line answer in simple words, with our actual numbers where relevant>
```

### 4.9 Writing style for all logs

- Write for a 2nd-year ECE student who is new to ML.
- Explain every technical term the first time it appears.
- Use short sentences.
- Use real numbers from our runs, never vague words like "good accuracy".

---

## 5. Environment notes

- **Where the notebooks run:** on Kaggle, with the NEU dataset attached and a GPU for training. `python kaggle/run_on_kaggle.py <notebook>` pushes a notebook, waits for the run and copies its outputs back (see `README.md`).
- **Kaggle CLI:** `pip install kaggle`, then sign in once with `kaggle auth login`. GPU and internet inside Kaggle notebooks need a phone-verified Kaggle account.
- **On a laptop instead:** Python 3.11 or newer, `pip install tensorflow pandas matplotlib scikit-learn pillow jupyter`, and the dataset unzipped so `data/raw/NEU-DET/` exists (see `data/README.md`). TensorFlow on native Windows is CPU-only, which is fine for the data and EDA notebooks.
- **Never print, log or commit API keys or tokens.**
