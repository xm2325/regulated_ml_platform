# Restore the complete source snapshot

The repository stores a checksum-verified source snapshot so the complete v0.5.0 tree can be recovered even when GitHub Actions is disabled by repository settings.

```bash
git clone https://github.com/xm2325/regulated_ml_platform.git
cd regulated_ml_platform
bash scripts/restore_source.sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make all
```

The restore script reconstructs the XZ archive, checks SHA-256 `0785e71b6a733c20be97adee4d75348d7fb5b88aefc531f717516057019756bf`, and extracts the complete source tree. `make all` then regenerates the 5,000-row synthetic dataset, processed feature table, model artifacts, batch predictions, governance reports, deployment validation, dashboard, and tests.
