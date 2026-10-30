# Things only the repository owner can do

Tick these before or right after the first push.

## Before pushing
- [ ] **Revoke the old Hugging Face token** that was hard-coded in the original scripts
      (<https://huggingface.co/settings/tokens>). Also rotate the token in your local `.env` if it was ever shared.
- [ ] Confirm `git status` is clean and `git ls-files` has **no** `.env`, videos, audio, or `eval/data/` files:
      `git ls-files | grep -E "\.(env|mp4|wav|mp3|mkv)$|eval/data"` should print nothing.
- [ ] Skim `README.md` and `AUDIT.md` once for anything you do not want public (names, paths, opinions).
- [ ] Decide whether to keep `AUDIT.md` and `docs/PROGRESS.md` (they are candid working notes; fine for a research
      portfolio, optional otherwise).

## Create the repository (GitHub)
```bash
# in this folder (it has its own git repo, branch main)
gh repo create profanity-diarization-pipeline --public --source . --remote origin
git push -u origin main
```
Suggested description: *Offline profanity detection and speaker diarization for movie clips with Whisper and
pyannote, with an honest evaluation of the published method.*
Suggested topics: `whisper`, `pyannote`, `speaker-diarization`, `speech-to-text`, `profanity-detection`,
`content-moderation`, `offline`, `python`.

After the first push:
- [ ] Open the **Actions** tab and check the CI run. It has not run on GitHub yet.
- [ ] Replace `<repo-url>` in the README quickstart with the real clone URL.
- [ ] (Optional) add the repo link to your profile README, CV, LinkedIn and the paper's author profile.

## Things that make the results stronger (not required to publish)
- [ ] **Audit the model-assigned labels** in `eval/data/*.labels.json` and rerun `make eval`; update the numbers in the
      README from the regenerated `eval/results.md`.
- [ ] **Hand-label speaker gender** in `eval/data/gender_template.csv` (M/F per turn, ~30 turns is enough) and run
      `python -m eval.gender_eval score`.
- [ ] Run the Docker image once (`docker compose run --rm profanity-diarization prefetch-models`, then `up`); the
      build was not tested during development.
- [ ] Record a short terminal GIF of a run and add it as `docs/demo.gif` (reference it in the README only once the
      file exists).
- [ ] Tag a release only after you are happy with it (`git tag v0.1.0`). The fidelity check against the original
      script has not been run, so avoid describing the release as "paper-exact" until it is.

## Licensing notes
- Code: MIT (see `LICENSE`). Confirm the lexicon is your own work (it is described as such in `data/lexicon/README.md`).
- Do not add the test videos. See `docs/DATA.md`.
