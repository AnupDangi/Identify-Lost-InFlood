# Privacy cleanup: committed demo screenshots

## What was found

`demo/images/image1.png` and `demo/images/image2.png` were committed to git in
commit `2a86f1e` ("feat:add demo images as well"). Per the README at the time,
these were screenshots of the review UI's **Dead Bodies** tab -- which, per
`ARCHITECTURE.md` and `docs/project_requirement.md`, renders real photographs
scraped from the Nepal Police UDB `dead-bodies-lists` endpoint (PM / unidentified
body records), along with the record's location, dates, and other metadata fields.
That makes the screenshots real, identifiable personal data of deceased individuals,
not synthetic/redacted examples.

These files were **not opened or inspected as part of this cleanup** -- the
classification above is based on the caption text ("Dead Bodies tab, candidate view")
and the surrounding documentation, which is sufficient to treat them as sensitive
without needing to view the contents.

## What was changed (this session, on `main`)

1. `git rm --cached demo/images/image1.png demo/images/image2.png` -- untracked the
   files going forward. The files still exist on disk locally (nothing was deleted
   from the working tree's filesystem beyond git's index); they are simply no longer
   part of the repository from this commit onward.
2. `.gitignore` now ignores `demo/images/`, `demo/private/`, `demo/raw/`, and
   `demo/real/`, so a future screenshot dropped into any of those paths will not be
   committed by accident.
3. `README.md`'s Demo section no longer references the removed images; it explains
   why and points here.

**This does NOT remove the images from git history.** `2a86f1e` (and every commit/tag
that has it as an ancestor, including the current `main` tip and whatever is on
`origin/main`) still contains the blobs. Anyone who clones the repo or has already
fetched it can still recover the files from history even though `git status` on a
fresh checkout of the new commit will show them absent from the working tree.

## If full removal from history is required

This was intentionally **not** done automatically -- it rewrites commit hashes for
every commit at or after `2a86f1e` (in this repo, that's just `2a86f1e` itself, since
it's currently the tip of `main`), and if this has already been pushed and pulled
elsewhere, it requires force-pushing and everyone else re-cloning or hard-resetting.
Only the repository owner should decide when to do this, ideally before anyone else
has pulled the current `main` tip.

Recommended tool: [`git-filter-repo`](https://github.com/newren/git-filter-repo)
(not the deprecated `git filter-branch` / BFG for this case, since filter-repo is the
officially recommended replacement and handles this cleanly).

```bash
# 1. Install (macOS): brew install git-filter-repo
# 2. From a FRESH clone (filter-repo refuses to run in a repo with unpushed work
#    or that isn't a fresh clone, to avoid surprising you):
git clone <this-repo-url> find-lost-inflood-history-cleanup
cd find-lost-inflood-history-cleanup

# 3. Strip the two blobs from every commit in history:
git filter-repo --path demo/images/image1.png --path demo/images/image2.png --invert-paths

# 4. Verify they're gone:
git log --all --oneline -- demo/images/image1.png demo/images/image2.png
#   -> should print nothing

# 5. Force-push the rewritten history (THIS REWRITES origin/main -- coordinate
#    with anyone else who has this repo cloned; they will need to re-clone or
#    hard-reset to the new history, not merge/pull):
git push origin --force --all
git push origin --force --tags
```

If the repo has ever been forked, mirrored, or cloned by someone outside your
control, treat the images as permanently public regardless of history rewriting --
`git filter-repo` only cleans the repositories it's run against.

## Going forward

- Never commit a screenshot, log excerpt, or test fixture that renders real AM/PM
  photos, names, case IDs, phone numbers, or other fields sourced from `data/`.
- `data/` itself was already gitignored before this session and remains the only
  place real scraped records should live.
- If a demo screenshot is needed for documentation, generate it against synthetic
  placeholder records (fake name, placeholder image, made-up IDs) so a screenshot of
  the actual UI carries no real personal data.
