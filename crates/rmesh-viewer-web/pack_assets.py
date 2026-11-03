"""Package the desktop torch's existing glTF and license for static web hosting."""
import base64
import json
import mimetypes
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
source = root / 'assets/flare'
output = Path(__file__).resolve().parent / 'web/assets'
output.mkdir(parents=True, exist_ok=True)
data = json.loads((source / 'scene.gltf').read_text())
for entry in data.get('buffers', []) + data.get('images', []):
    uri = entry.get('uri', '')
    if uri and not uri.startswith('data:'):
        mime = mimetypes.guess_type(uri)[0] or 'application/octet-stream'
        entry['uri'] = 'data:' + mime + ';base64,' + base64.b64encode((source / uri).read_bytes()).decode()
(output / 'flare.gltf').write_text(json.dumps(data, separators=(',', ':')))
shutil.copyfile(source / 'license.txt', output / 'flare-license.txt')

shutil.copyfile(root / 'crates/rmesh-anim/tests/fixtures/slide.gltf', output / 'animation-test.gltf')
