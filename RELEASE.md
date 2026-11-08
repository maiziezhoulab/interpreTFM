# Release checklist

Steps for a tagged release (`vX.Y.Z`) of interpreTFM. Each step names the file or service it touches; the person releasing runs them in order.

## 1. Version and metadata

1. Set `__version__` in `src/interp_pipeline/__init__.py` (the package version is read from it) and remove the `.dev` suffix.
2. `CITATION.cff`: `version`, and `date-released` (`YYYY-MM-DD`); the software author list stays as it is (Yunfei Hu). When the manuscript is published, add its DOI under `preferred-citation`. The release branch may carry the literal `RELEASE_DATE` in `date-released` and in the CHANGELOG heading; it is replaced by the release date in the release commit.
3. `CHANGELOG.md`: rename the `[X.Y.Z] – unreleased` heading to `[X.Y.Z] - YYYY-MM-DD`, check that every entry describes released behaviour, and update the compare link at the bottom (`[X.Y.Z]: .../compare/v<previous>...vX.Y.Z`, plus a new `[Unreleased]` link).
4. `README.md`: the badges and the quick start reflect the release; no "in preparation" wording that the release makes false.
5. `envs/*.yml`, `containers/`, `docs/requirements.txt`: unchanged unless a dependency moved.

## 2. Checks before tagging

```bash
ruff check .
python -m pytest                                   # unit tests; reproduction tests skip without the data
sphinx-build -W -b html docs docs/_build/html      # documentation, warnings are errors
git grep -n -i -E '<private path fragments>'       # nothing machine-specific in tracked files
interpretfm hub verify <run_dir…> <export_dir>     # the exported SAEs reload bit-identically
```

A fresh environment from `envs/interpretfm.yml` must install the package and pass the tests (`python -m pip install -e . && python -m pytest`).

## 3. Tag and GitHub release

```bash
git checkout main && git pull
git tag -a vX.Y.Z -m "interpreTFM X.Y.Z"
git push origin main vX.Y.Z            # push the release tag by name, not --tags
```

On GitHub, create a release from the tag with the CHANGELOG section as its text. The CI workflow runs on the push; the release is final only when it is green.

## 4. Zenodo (software DOI)

1. Enable the repository once under <https://zenodo.org/account/settings/github/> (GitHub integration, repository switch on).
2. Every GitHub *release* (not a bare tag) is then archived automatically and gets a DOI; the concept DOI stays the same across versions.
3. Add the version DOI to `CITATION.cff` (`identifiers: - type: doi`) and to the README citation block in a follow-up commit; the metadata Zenodo shows come from `CITATION.cff` (title, authors, license, keywords).

## 5. Read the Docs

1. Import the repository once at <https://readthedocs.org/dashboard/import/> (project name `interpretfm`, so that the badge and links in the README resolve); the build uses `docs/conf.py` and `docs/requirements.txt`.
2. Activate the tagged version under *Versions* and set it as the default when the release is final; the `latest` build follows the default branch.
3. Trigger a rebuild from the dashboard after the tag (automatic when the webhook is installed).

## 6. Container images

The `Containers` workflow (`.github/workflows/containers.yml`) is started by hand (`workflow_dispatch`): Actions tab → Containers → *Run workflow*, choose the images (`interpretfm,geneformer,c2s`), the tag (the release version) and `push=true` to publish them to `ghcr.io/<owner>/interpretfm-<name>:<tag>`. A workflow appears in the Actions tab only when its file exists on the default branch, so merge first, then start it. Make the GHCR packages public after the first push (package settings on GitHub).

## 7. Hugging Face: trained SAEs and the tutorial bundle

1. Export the finished run directories (one folder per model, dataset and layer, with the model cards): `interpretfm hub export <run_dir…> <export_dir> --repo-id YunfeiHu/interpreTFM`, then `interpretfm hub verify <run_dir…> <export_dir>`.
2. Put the tutorial bundle beside the export: `<export_dir>/tutorial/` (the projected features, the store slice, the label groups and its `README.md`; no CosMx crop: the tutorial builds it with `interpretfm.hub.tutorial_crop` from the public source); the tutorial loads it with `interpretfm.hub.tutorial_bundle("hub", repo_id=...)`.
3. Create the model repository and upload (needs `huggingface_hub` and a write token, `huggingface-cli login`):

   ```bash
   huggingface-cli repo create interpreTFM --type model
   huggingface-cli upload YunfeiHu/interpreTFM <export_dir> . --repo-type model --commit-message "interpreTFM X.Y.Z SAEs and tutorial bundle"
   huggingface-cli tag YunfeiHu/interpreTFM vX.Y.Z                      # the version tag of the Hub repository
   ```

4. The repository id is `interpretfm.hub.DEFAULT_REPO` (`YunfeiHu/interpreTFM`), used by the tutorial notebook (`REPO_ID`) and the model cards (`--repo-id`); a new repository means changing it there and re-exporting.
5. Check the licence line of every model card (GPL-3.0-or-later for the SAEs; the base models and datasets keep their own terms, see `THIRD_PARTY_NOTICES.md`).

## 8. After the release

- Bump `__version__` to the next `.dev0`, add an `[Unreleased]` section to the CHANGELOG.
- Record the DOIs (Zenodo, manuscript) in `CITATION.cff` and the README.
