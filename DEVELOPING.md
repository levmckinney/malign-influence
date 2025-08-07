### Developer setup with uv and Docker

This repo is configured to use uv for dependency management. You can develop either:

- Locally on your host with a virtual environment
- Inside Kubernetes via the provided SSH dev box (port-forward)

#### 1) Local (host) development

Requirements: `uv` installed on your machine. Install via the official script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # if not already on PATH
```

Create a venv and install deps:

```bash
cd $(git rev-parse --show-toplevel)
uv sync --all-groups   # or: uv sync --frozen --all-groups

# Run tools/commands inside the project venv via uv
uv run python -V
uv run pytest -q
```

Common commands:

- Install a new dep: `uv add PACKAGE`
- Update locks: `uv lock`
- Export pinned requirements: `uv export --frozen > requirements.txt`

#### 2) Kubernetes dev box (SSH via port-forward)

Build the devbox image and make it available to your cluster:

```bash
docker build -t malign/devbox:latest --target devbox .
# Push to a registry your cluster can pull from, e.g. Docker Hub
# docker push <registry-namespace>/malign/devbox:latest
# OR, for kind: kind load docker-image malign/devbox:latest
```

Apply the manifest and add your SSH key:

```bash
kubectl apply -f k8s/devbox.yaml
# Create the Secret containing your SSH public key (do not commit keys to git)
kubectl -n devbox create secret generic ssh-authorized-keys \
  --from-file=authorized_keys=$HOME/.ssh/id_ed25519.pub \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n devbox rollout status deploy/devbox
```

Port-forward the SSH service locally and connect:

```bash
kubectl -n devbox port-forward svc/devbox-ssh 2222:2222
ssh -p 2222 dev@localhost
```

Inside the box:

```bash
cd /workspace
uv sync --all-groups
uv run pytest -q
```

#### Production/runtime image

The default `runtime` target is a minimal Python image containing only the app and its virtual environment. Build it with:

```bash
docker build -t malign/app:latest --target runtime .
```

Run it (override CMD as needed):

```bash
docker run --rm -it malign/app:latest bash
```

#### Need CUDA/GPU?

This setup uses standard Debian slim images for portability. If you require CUDA/PyTorch GPU, we can add a GPU-specific base and target that follow the same uv pattern. Let me know your desired CUDA/PyTorch versions.


