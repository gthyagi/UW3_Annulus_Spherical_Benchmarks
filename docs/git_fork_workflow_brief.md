# Git Fork Workflow (Quick Reference)

## Remotes

- `upstream` = official repo
- `origin` = your fork

Flow:

`upstream -> local -> origin`

## One-Time Setup

```bash
git clone https://github.com/gthyagi/underworld3.git
cd underworld3
git remote add upstream https://github.com/underworldcode/underworld3.git
git fetch upstream
git switch -c development upstream/development
git push -u origin development
```

## Sync `development`

```bash
git switch development
git fetch upstream
git pull --ff-only upstream development
```

## Start New Work

Use `feature/...` or `bugfix/...`, not `codex/...`.

```bash
git switch development
git pull --ff-only upstream development
git switch -c feature/my-change
```

## Commit And Push

```bash
git status
git add <files>
git commit -m "Short message"
git push -u origin feature/my-change
```

## If You Rebased

```bash
git push --force-with-lease origin feature/my-change
```

## Useful Checks

```bash
git branch -vv
git remote -v
git log --oneline --graph --decorate --all
git diff --stat
```

## Build Note

Use:

```bash
pip install . --no-build-isolation
```

That tells `pip` to build the package without creating an isolated temporary build environment.

## In Pixi Env

Activate the env first:

```bash
pixi shell -e amr-dev
pip install . --no-build-isolation
```

Or run it directly:

```bash
pixi run -e amr-dev pip install . --no-build-isolation
```

```bash
OMPI_CC=/usr/bin/clang \
OMPI_CXX=/usr/bin/clang++ \
pixi run -e amr-dev pip install . --no-build-isolation
```
```bash
OMPI_CC=/usr/bin/clang \
OMPI_CXX=/usr/bin/clang++ \
./uw build
```

```bash
OMPI_CC=/usr/bin/clang \
OMPI_CXX=/usr/bin/clang++ \
./uw test
```

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export OMPI_CC=/usr/bin/clang
export OMPI_CXX=/usr/bin/clang++
```