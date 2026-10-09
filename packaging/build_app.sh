#!/bin/sh
# Build dist/firstlight.app (a double-clickable Mac app). Needs: pip install -e '.[package]'
# Siril is not bundled: install it with `brew install siril` or from siril.org; the app looks in the usual places.
set -e
cd "$(dirname "$0")/.."
.venv/bin/pyinstaller --noconfirm --windowed --name firstlight \
  --osx-bundle-identifier com.firstlight.app \
  --add-data "firstlight/app_ui.html:firstlight" \
  --add-data "firstlight/widget_template.html:firstlight" \
  --add-data "firstlight/widget_dsp.js:firstlight" \
  --paths . --collect-submodules firstlight --collect-submodules webview --additional-hooks-dir packaging/hooks \
  --exclude-module playwright --exclude-module pytest --exclude-module tkinter \
  packaging/launch.py
echo "built dist/firstlight.app"
