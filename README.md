# Typhoid Prediction in Nepal — Research Site

A living research site for *Predictive Modeling of Typhoid Incidence in
Nepal Under Extreme Climate Change Scenarios Using Machine Learning*
(Kritika Baral, Kathmandu University). Paper text, methodology,
experiments, figures, replication code, and a research presentation —
all in one place, all rebuilt from source on every push.

Built with [Quarto](https://quarto.org). Hosted free on GitHub Pages.

---

## Table of contents

1. [What you get](#what-you-get)
2. [Prerequisites — install once](#prerequisites--install-once)
3. [Run the site locally](#run-the-site-locally)
4. [Site structure](#site-structure)
5. [Editing content](#editing-content)
6. [Publish on GitHub Pages — free hosting](#publish-on-github-pages--free-hosting)
7. [Updating after first publish](#updating-after-first-publish)
8. [Optional: custom domain](#optional-custom-domain)
9. [Troubleshooting](#troubleshooting)
10. [Data and trained models](#data-and-trained-models)
11. [Why Quarto](#why-quarto)
12. [License](#license)

---

## What you get

- A **multi-page website** with Home · Notebook · Presentation slides · Paper
  (5 chapters) · Replication · Figures · About.
- A **paper PDF** rendered from the same source (`make pdf`).
- A **research presentation** in two formats: in-browser revealjs slides
  *and* a downloadable `.pptx`.
- A **runnable replication pipeline** — every figure and table in the
  paper regenerates from code.
- **APA in-text citations** that hover-preview and click-jump-and-highlight
  the corresponding bibliography entry.
- **Free hosting** on GitHub Pages — costs nothing, no servers, no
  Python runtime on the visitor's side. Everything renders to static HTML
  at build time.

---

## Prerequisites — install once

You need three tools on your laptop. After this you never reinstall.

### 1. Git

```bash
# macOS — comes with Xcode Command Line Tools
xcode-select --install

# Verify
git --version
```

### 2. Python 3.11+

```bash
# macOS — recommended via Homebrew
brew install python@3.11

# Verify
python3 --version
```

### 3. Quarto

Download the installer from <https://quarto.org/docs/get-started/> — or:

```bash
# macOS
brew install --cask quarto

# Verify
quarto --version       # should be ≥ 1.5
```

### 4. (Optional) TinyTeX — only if you want PDF output

Quarto needs LaTeX to render `make pdf`. The easiest path is the
Quarto-managed TinyTeX install:

```bash
quarto install tinytex
```

You can skip this if you only care about the website.

---

## Run the site locally

```bash
# Clone (or download) the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# One-time: create the Python virtualenv and install deps (~1 minute)
make install

# Live-reloading browser preview — opens http://localhost:<port> automatically
make preview

# OR: a single full render to ./_site/  (no live reload)
make

# Also render the paper as a standalone PDF (requires TinyTeX)
make pdf

# Wipe rendered output + caches if something goes weird
make clean
```

What `make preview` does step by step:

1. Activates the `.venv/` Python virtualenv.
2. Runs `quarto preview`, which:
   - Executes every `{python}` cell in every `.qmd` (first time only — cached after).
   - Renders all pages to HTML.
   - Spins up a local web server.
   - Opens your default browser to the site.
   - Watches every `.qmd` / `.css` / `_quarto.yml` for changes and re-renders on save (~1 second).

The first render is slow (10–60 seconds) because Quarto has to download Google Fonts, start a Jupyter kernel, and execute all code cells. After that, the `_freeze/` cache makes subsequent renders almost instant.

---

## Site structure

```
.
├── _quarto.yml                # site config: title, nav, theme, bibliography
├── index.qmd                  # landing page
├── about.qmd                  # how-it's-built
├── notebook.qmd               # single-page research walkthrough
├── presentation.qmd           # presentation slides (revealjs + pptx output)
├── presentation.css           # slide-specific styling
├── figures.qmd                # all 5 figures with captions
├── paper/                     # the write-up — chapter per .qmd
│   ├── 01-introduction.qmd
│   ├── 02-literature-review.qmd
│   ├── 03-methodology.qmd
│   ├── 04-results.qmd         # embedded Python cells produce the headline tables
│   └── 05-discussion.qmd
├── replication/               # runnable, end-to-end reproduction
│   ├── data.qmd
│   ├── train.qmd
│   └── eval.qmd
├── data/                      # raw CSVs (small, committed)
├── tables/                    # pre-computed result tables (CSV)
├── figures/                   # publication figures (PNG)
├── refs.bib                   # BibTeX, APA-cited as [@key]
├── styles.css                 # site-wide theme overrides
├── requirements.txt           # Python deps, version-pinned
├── Makefile                   # install / preview / render / publish / clean
├── .gitignore
└── .github/workflows/
    └── publish.yml            # CI: auto-publishes to GitHub Pages on every push
```

---

## Editing content

- Open any `.qmd` in a text editor (VS Code, Cursor, Zed, Vim — anything).
- Format is **Markdown** + a YAML front-matter block + optional `{python}` code cells.
- Save the file. If `make preview` is running, the browser auto-refreshes within ~1 second.
- Code cells execute at build time; outputs (charts, tables, printed values) are baked into the HTML. No server-side Python at visit time.
- Cite as `[@key]` → renders as `(Author, Year)`. Hover any citation for the full reference; click it to jump-and-highlight on the References list.
- New citations: add an entry to `refs.bib` in APA-friendly BibTeX format, then cite with `[@yourkey]`.
- New images: drop into `figures/` and reference as `![caption](figures/foo.png)`.
- New data: drop a CSV in `data/` and load in any code cell with `pd.read_csv("data/your.csv")`.

---

## Publish on GitHub Pages — free hosting

This is a one-time setup. After it works, future updates publish automatically on every `git push`.

### Step 1 — Create a GitHub account (if you don't have one)

Sign up at <https://github.com/join>. Free.

### Step 2 — Create a new repository

1. Go to <https://github.com/new>.
2. **Repository name**: something memorable, e.g. `typhoid-nepal-research` or `kritikabaral.github.io`.
   - **Special case**: if you name the repo `<username>.github.io` (e.g. `kritikabaral.github.io`), the site URL is `https://<username>.github.io/` (cleaner, no `/repo-name/` suffix).
   - Otherwise the URL is `https://<username>.github.io/<repo-name>/`.
3. **Visibility**: Public (required for GitHub Pages on the free plan).
4. **Do NOT** initialize with a README, .gitignore, or license — you already have those.
5. Click **Create repository**.

### Step 3 — Push your local project to GitHub

In your terminal, inside the project folder:

```bash
# Initialize git if you haven't (skip if already a repo)
git init
git add .
git commit -m "Initial commit — research site scaffold"

# Tell git where the remote is — replace USERNAME and REPONAME
git branch -M main
git remote add origin https://github.com/USERNAME/REPONAME.git

# Push to GitHub
git push -u origin main
```

If git prompts for credentials, GitHub no longer accepts your password —
you need a **Personal Access Token** (PAT) or to set up an SSH key.

- Easiest: install the GitHub CLI (`brew install gh`), run `gh auth login`, follow the prompts, then re-run `git push`.
- Or follow GitHub's guide: <https://docs.github.com/en/authentication>.

### Step 4 — Enable GitHub Pages

1. In your repo on GitHub, click **Settings** (gear icon, top-right of the repo header).
2. In the left sidebar, click **Pages**.
3. Under **Build and deployment** → **Source**, select **Deploy from a branch**.
4. Under **Branch**, select **gh-pages** and folder **/ (root)**. Click **Save**.
   - **Note**: the `gh-pages` branch doesn't exist yet — it'll be created automatically by the first workflow run. You may need to come back to this step after Step 6.

### Step 5 — Grant the workflow write permission

The publish workflow needs permission to push to the `gh-pages` branch.

1. Still in **Settings**, sidebar → **Actions** → **General**.
2. Scroll to **Workflow permissions** (bottom of the page).
3. Select **Read and write permissions**. Click **Save**.

### Step 6 — Trigger the first deploy

Either:

```bash
# Make any tiny edit, commit, push — triggers the workflow
echo "" >> README.md
git add README.md
git commit -m "Trigger first deploy"
git push
```

…or click **Actions** tab in the GitHub repo UI → select the **publish** workflow → click **Run workflow** → **Run workflow** (manual dispatch).

### Step 7 — Watch it build

1. Click the **Actions** tab in your GitHub repo.
2. You'll see a workflow run called **publish**. Click it.
3. Watch the steps: Checkout → Set up Python → Install deps → Install Quarto → Render → Publish to GitHub Pages. Takes 2–4 minutes the first time.
4. When the run completes (green check), refresh the **Settings → Pages** screen. GitHub will display the live URL: `https://USERNAME.github.io/REPONAME/`.
5. Visit it in your browser.

### Step 8 — Replace every placeholder

This is the **one mandatory pass** before your site looks finished.
There are exactly four placeholder strings in the source — each one is
listed below with its file, line, what it currently says, and what you
must change it to.

In every case, replace:

- `your-username` (or `USERNAME`) → your actual GitHub username
- `your-repo` (or `REPONAME`) → your actual repository name (the part
  after the `/` in the GitHub repo URL)

Use a text editor (VS Code, Cursor, etc.) and find-and-replace, or open
each file by hand:

---

#### Placeholder 1 — Navbar GitHub icon

- **File**: [`_quarto.yml`](_quarto.yml)
- **Line**: 31
- **Current**:
  ```yaml
      right:
        - icon: github
          href: https://github.com/your-username/your-repo
  ```
- **Change to**: your real repo URL, e.g.
  `https://github.com/kritikabaral/typhoid-nepal-research`
- **Why it matters**: the GitHub icon in the top-right of the navbar
  on every page links here. Wrong URL → readers can't find your code.

---

#### Placeholder 2 — Page footer GitHub icon

- **File**: [`_quarto.yml`](_quarto.yml)
- **Line**: 53
- **Current**:
  ```yaml
      right:
        - icon: github
          href: https://github.com/your-username/your-repo
  ```
- **Change to**: same as Placeholder 1.
- **Why it matters**: same icon appears at the bottom of every page.
  Keep both in sync.

---

#### Placeholder 3 — "How to cite" URL on the home page

- **File**: [`index.qmd`](index.qmd)
- **Line**: ~90
- **Current**:
  ```markdown
  > URL: https://your-username.github.io/your-repo
  ```
- **Change to** (one of these, depending on Step 2 earlier):
  - If your repo is named `username.github.io`:
    `https://kritikabaral.github.io/`
  - Otherwise:
    `https://kritikabaral.github.io/typhoid-nepal-research/`
  - If you set up a custom domain (see [Optional: custom domain](#optional-custom-domain)):
    `https://yourdomain.com/`
- **Why it matters**: this is the URL anyone copy-pastes when citing
  your work in their own paper.

---

#### Placeholder 4 — README link in "About" page

- **File**: [`about.qmd`](about.qmd)
- **Line**: ~26
- **Current**:
  ```markdown
  See the [README](https://github.com/your-username/your-repo#readme) for the
  ```
- **Change to**:
  `https://github.com/kritikabaral/typhoid-nepal-research#readme`
- **Why it matters**: visitors who want to clone and build the site
  follow this link to the build instructions.

---

#### Optional but recommended: author + affiliation

If you're forking this for a different project, also update:

- [`index.qmd`](index.qmd) (front-matter) — `author.name`, `author.orcid`,
  `author.affiliations`, `author.email`. Currently set to Kritika Baral
  / Kathmandu University.
- [`presentation.qmd`](presentation.qmd) (front-matter) — same fields.
- [`_quarto.yml`](_quarto.yml) — `website.title`, `website.description`,
  `page-footer.left` (currently `© 2026 Samrat Baral`).

---

#### Verify nothing was missed

After your replacements, run:

```bash
grep -rn "your-username\|your-repo\|USERNAME\|REPONAME" \
  --include="*.qmd" --include="*.yml" --include="*.md" .
```

If the only hits are inside `README.md` (which uses these strings as
documentation examples), you're done.

### Step 9 — Commit and push your changes

```bash
git add _quarto.yml index.qmd about.qmd
git commit -m "Replace placeholder URLs with real repo info"
git push
```

GitHub Actions kicks off another build. In ~3 minutes, your live site
shows the real URLs everywhere.

---

## How to change content on the website (the right way)

**Your live site is a build artifact.** Don't edit it directly on
GitHub. Always:

1. Edit the source `.qmd` / `.yml` / `.css` / `.bib` file **locally**.
2. Preview the change with `make preview`.
3. Commit + push when you're happy.
4. GitHub Actions rebuilds and redeploys automatically.

This loop is the *only* way changes reach the live site. Here's the
detailed workflow for the most common edits:

### Change the abstract or any paragraph of text

1. Open the relevant `.qmd` in your editor:
   - Abstract → [`index.qmd`](index.qmd) (in the YAML front-matter under `abstract:`)
   - Introduction → [`paper/01-introduction.qmd`](paper/01-introduction.qmd)
   - Methodology → [`paper/03-methodology.qmd`](paper/03-methodology.qmd)
   - etc.
2. Edit the prose. Save.
3. If `make preview` is running, the browser refreshes within ~1s.
4. Commit + push:
   ```bash
   git add paper/01-introduction.qmd
   git commit -m "Tighten phrasing in intro motivation"
   git push
   ```

### Update a number in the results

1. Open [`tables/performance_metrics.csv`](tables/performance_metrics.csv)
   or [`tables/lag_correlation.csv`](tables/lag_correlation.csv).
2. Edit the CSV (any editor — even Excel/Numbers, but save as plain CSV).
3. `make preview` — every page that loads this CSV (Results, Eval,
   Notebook) updates automatically.
4. Commit + push.

### Add or replace a figure

1. Drop the new PNG into [`figures/`](figures/) (small, < 1 MB ideally).
2. Reference it in the `.qmd` that needs it:
   ```markdown
   ![Your caption here.](figures/your-new-figure.png){#fig-yours width=80%}
   ```
3. Cross-reference elsewhere with `@fig-yours`.
4. Commit + push.

### Add a new citation

1. Open [`refs.bib`](refs.bib).
2. Add a BibTeX entry. The key is your in-text handle (no spaces, lowercase):
   ```bibtex
   @article{smith2024,
     author  = {Smith, Jane and Doe, John},
     title   = {Title of paper},
     journal = {Journal Name},
     volume  = {12},
     number  = {3},
     pages   = {45--67},
     year    = {2024},
     doi     = {10.xxxx/yyy}
   }
   ```
3. Cite anywhere with `[@smith2024]` → renders as `(Smith & Doe, 2024)`.
4. Commit + push — the References list on the citing page auto-updates.

### Rename or add a new page

1. Create a new `.qmd` file (e.g. `methods-appendix.qmd`).
2. Add the YAML front-matter:
   ```yaml
   ---
   title: "Methods Appendix"
   ---
   ```
3. Register it in the navbar / sidebar via [`_quarto.yml`](_quarto.yml):
   ```yaml
   website:
     navbar:
       left:
         - href: methods-appendix.qmd
           text: Appendix
   ```
4. Commit + push.

### Change the site theme or colors

- **Slides** — edit [`presentation.css`](presentation.css) (CSS variables at the
  top under `:root`).
- **Web pages** — edit [`styles.css`](styles.css) (CSS variables at the
  top under `:root`).
- **Bootstrap theme** — change `format.html.theme` in [`_quarto.yml`](_quarto.yml)
  (currently `cosmo` light / `darkly` dark; full list at <https://quarto.org/docs/output-formats/html-themes.html>).

### Change the citation style

- **Currently APA 7th edition**.
- To switch: edit [`_quarto.yml`](_quarto.yml), change the `csl:` line:
  ```yaml
  # Default — APA
  csl: https://www.zotero.org/styles/apa

  # IEEE
  csl: https://www.zotero.org/styles/ieee

  # Vancouver (medical journals)
  csl: https://www.zotero.org/styles/vancouver

  # Chicago author-date
  csl: https://www.zotero.org/styles/chicago-author-date
  ```
- Full list of styles: <https://www.zotero.org/styles>

### Roll back a bad change after pushing

Use git:

```bash
# See the last few commits
git log --oneline -5

# Revert a specific commit (creates a new commit that undoes it)
git revert <commit-hash>
git push
```

GitHub Pages auto-rebuilds from the new commit within minutes.

---

## Updating after first publish

After initial setup, daily workflow is just:

```bash
# Edit any .qmd locally — preview as you go
make preview

# When happy, commit + push — site auto-rebuilds and deploys in ~3 min
git add .
git commit -m "Describe what you changed"
git push
```

That's it. Watch the **Actions** tab on GitHub to see the build progress.

---

## Optional: custom domain

If you own a domain (e.g. `kritikabaral.com`) and want to use it instead of `github.io`:

1. In your domain registrar's DNS settings, add:
   - An **A record** pointing your apex domain (`@`) to GitHub's IPs:
     ```
     185.199.108.153
     185.199.109.153
     185.199.110.153
     185.199.111.153
     ```
   - Or a **CNAME record** for `www` pointing to `USERNAME.github.io`.
2. In GitHub repo **Settings → Pages → Custom domain**, enter your domain and Save.
3. Check **Enforce HTTPS** once the certificate has been provisioned (5–30 min).
4. GitHub will create a `CNAME` file in your repo automatically.

Detailed walkthrough: <https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site>

---

## Troubleshooting

### `make preview` fails with `quarto: command not found`

You skipped the Quarto install. Run `brew install --cask quarto` (macOS) or download from <https://quarto.org/docs/get-started/>.

### `ModuleNotFoundError` when rendering

The virtualenv is missing a package. Add it to `requirements.txt`, then:

```bash
make clean
make install
make preview
```

### The site builds locally but fails on GitHub Actions

Check the **Actions** tab → click the failed run → expand the failed step. Common causes:

- **Permissions**: re-check Step 5 above (Workflow permissions = Read and write).
- **Python version mismatch**: the workflow uses Python 3.11. If you're using something else locally, things should still match, but if you've added a 3.12-only dep, pin a 3.11-compatible version.
- **Pickle / cache file in repo**: don't commit `_site/`, `_freeze/`, or `.venv/`. They're in `.gitignore` — make sure you didn't override.

### Warning: "Unable to resolve link target"

You've got a stale link to a renamed file. Search the codebase:

```bash
grep -rn "old-filename.qmd" .
```

…and update the links. If the source is fine but the warning persists, your execution cache is stale:

```bash
rm -rf _freeze/<path-to-page> .quarto/_freeze/<path-to-page>
make preview
```

### Site is up but pages 404

GitHub Pages takes 1–5 minutes to propagate after the first successful deploy. Wait a bit, then hard-refresh (Cmd+Shift+R / Ctrl+Shift+R).

### Citation tooltip doesn't appear on hover

Make sure `_quarto.yml` has `citations-hover: true` under `format.html` (it does, by default in this repo). Hard-refresh the browser to bust the cache.

---

## Data and trained models

- **Raw CSVs** are committed to `data/`: `typhoid_data_1.csv`, `typhoid_data_2.csv`, `climate_data.csv`, `flood_data.csv`. All small (< 400 KB each).
- **Pre-computed tables** in `tables/`: `performance_metrics.csv`, `descriptive_statistics.csv`, `lag_correlation.csv`. Loaded by `paper/04-results.qmd` and `replication/eval.qmd`.
- **Figures** in `figures/`: 5 PNGs generated upstream by `visualizer.py`. Embedded into `figures.qmd` and referenced from Results.
- **Trained models** (`rf_model.pkl`, `xgb_model.pkl`, `mlp_model.pkl`, scalers) are **not** in this repo — `rf_model.pkl` is 16 MB. They live in the upstream Python project and are regenerable by running `python main.py` there.

### Re-running the production pipeline

```bash
cd <path-to-upstream-pipeline>
pip install -r requirements.txt
python main.py
```

Then copy refreshed CSVs into `tables/` and PNGs into `figures/`, and `make preview` to confirm.

---

## Why Quarto

- **One source, multiple outputs.** Same `.qmd` files render to a website AND a paper PDF AND presentation slides (revealjs + pptx).
- **Live code.** Python cells execute at build time — every chart matches the code that produced it.
- **Academic primitives.** LaTeX equations, BibTeX/APA citations, cross-references, callouts, themed code — all first-class.
- **Markdown, not Word.** Edit in any text editor, diff in git, no formatting drift.
- **Free hosting.** Static HTML output ships easily to GitHub Pages, Vercel, Netlify — any static host.

---

## License

Choose a license for the writing (e.g. **CC BY 4.0**) and another for the code (e.g. **MIT** or **Apache 2.0**). Add as `LICENSE-content` and `LICENSE-code` in the repo root.

For typical academic publishing: CC BY 4.0 lets others share and adapt your writing with attribution; MIT lets others use your code freely.
