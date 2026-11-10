"""Apply a guarded, backed-up adapter to an existing S v0.6 installation.

Stop S before applying; restart it afterwards. This does not deploy older Continuum
code from the research branch over the installed application or alter its database.
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def replacements(root):
    changes = {}
    anchors = {
        'continuum/engine.py': [("    if model==S_RESEARCH_MODEL:\n",
            "    if model=='s-native:local':\n"
            "        from s_intelligence.browser import complete\n"
            "        begin=time.monotonic()\n"
            "        try:response=complete(user_input)\n"
            "        except (ValueError,RuntimeError,OSError,ImportError) as e:raise LocalModelError(str(e)) from e\n"
            "        elapsed=int((time.monotonic()-begin)*1000)\n"
            "        item=store.add_message(sid,'assistant',response,model=model,elapsed_ms=elapsed)\n"
            "        store.log(pid,'chat',f'model={model} elapsed_ms={elapsed} verified_inference=native_neural')\n"
            "        return item\n"
            "    if model==S_RESEARCH_MODEL:\n", 1)],
        'continuum/server.py': [("models.extend(ollama_models())",
            "from s_intelligence.browser import available,MODEL_ID\n{indent}if available():models.append(MODEL_ID)\n{indent}models.extend(ollama_models())", 2)],
        'web/app.js': [("const modelCatalog={\n",
            "const modelCatalog={\n 's-native:local':{name:'S Native Lab',description:'Experimental neural text · ASCII · 32-character context · unreliable',symbol:'N'},\n", 1)],
    }
    for name, edits in anchors.items():
        path = root / name
        text = path.read_text(encoding='utf8')
        if 's-native:local' in text or 's_intelligence.browser' in text:
            raise ValueError('Native integration already exists; refusing to overwrite')
        for old, new, count in edits:
            if text.count(old) != count:
                raise ValueError(f'Unsupported source layout: {name}')
            if '{indent}' in new:
                lines = text.splitlines(keepends=True)
                text = ''.join(line.replace(old, new.replace('{indent}', line[:len(line)-len(line.lstrip())]))
                               if old in line else line for line in lines)
            else:
                text = text.replace(old, new)
        changes[name] = text
    return changes


def install(root):
    root = Path(root).resolve()
    changes = replacements(root)
    if (root / 's_intelligence').exists():
        raise FileExistsError('Preserve existing native package; review it before upgrading')
    evaluation = root / 'data/evaluation/native-reasoning.json'
    if evaluation.exists():
        raise FileExistsError('Preserve existing reasoning fixtures')
    checkpoint = root / 'native-research/S-INT-002/artifacts/s-int-002/steps-80-seed-20261110.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError('Install the verified S-INT-002 checkpoint first')
    backup = root / 'upgrade-backups' / ('native-browser-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir(parents=True)
    for name in changes:
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    shutil.copytree(ROOT / 's_intelligence', root / 's_intelligence', ignore=shutil.ignore_patterns('__pycache__'))
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'data/evaluation/native-reasoning.json', evaluation)
    for name, text in changes.items():
        (root / name).write_text(text, encoding='utf8')
    print(f'Native browser adapter installed; backup: {backup}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('installation', type=Path)
    install(parser.parse_args().installation)
