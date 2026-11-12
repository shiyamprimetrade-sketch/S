"""Checksum-bound local copy adapters; installation never establishes capability."""
import json
from pathlib import Path
import shutil
import tempfile
from s_brain.checkpoint import load_manifest,sha256_file


def validate_adapter(path,base_manifest):
    path=Path(path)
    weight=path/'adapter.safetensors'
    if weight.stat().st_size>2000000:raise ValueError('Copy adapter exceeds 2 MB budget')
    if (path/'manifest.json').stat().st_size>32000:raise ValueError('Copy manifest exceeds budget')
    value=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if value.get('schema')!=1 or value.get('kind') not in ('s-learned-byte-copy','s-learned-span-copy'):raise ValueError('Unsupported copy adapter schema')
    cfg=value.get('adapter_config',{})
    fields={'dim','hidden'} if value['kind']=='s-learned-span-copy' else {'dim','key_dim'}
    if set(cfg)!=fields or any(type(cfg[k]) is not int for k in cfg):raise ValueError('Invalid copy architecture')
    width=cfg.get('hidden',cfg.get('key_dim'))
    if not 1<=cfg['dim']<=512 or not 1<=width<=64:raise ValueError('Copy architecture exceeds budget')
    if cfg['dim']!=base_manifest.model_config['dim']:raise ValueError('Copy/base dimension mismatch')
    if value.get('base_weights_sha256')!=base_manifest.weights_sha256:raise ValueError('Copy adapter requires a different base checkpoint')
    if type(value.get('step')) is not int or value['step']<0:raise ValueError('Invalid copy training step')
    if sha256_file(weight)!=value.get('weights_sha256'):raise ValueError('Copy adapter checksum mismatch')
    return value


def load_adapter(path,base_manifest):
    value=validate_adapter(path,base_manifest)
    from safetensors.torch import load_file
    from s_brain.byte_copy import ByteCopy
    if value['kind']=='s-learned-span-copy':
        from s_brain.span_copy import SpanCopy
        adapter=SpanCopy(**value['adapter_config'])
    else:adapter=ByteCopy(**value['adapter_config'])
    adapter.load_state_dict(load_file(str(Path(path)/'adapter.safetensors')),strict=True)
    adapter.eval()
    return adapter,value


def install_copy_adapter(db,path):
    from .models import load_config,save_config
    config=load_config(db)
    if not config.get('native_checkpoint'):raise ValueError('Install the adapter\'s native base checkpoint first')
    base=load_manifest(Path(config['native_checkpoint']));value=validate_adapter(path,base)
    if value['step']<1:raise ValueError('Install a trained adapter; random control is evaluation-only')
    root=Path(db).resolve().parent/'s-copy-adapters'
    if root.is_symlink():raise ValueError('Symlinked copy model root')
    root.mkdir(parents=True,exist_ok=True);target=root/value['weights_sha256']
    if target.is_symlink():raise ValueError('Symlinked adapter target')
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            staged=Path(temporary)/'adapter';staged.mkdir()
            for name in ('manifest.json','adapter.safetensors'):shutil.copyfile(Path(path)/name,staged/name)
            validate_adapter(staged,base);staged.rename(target)
    validate_adapter(target,base)
    save_config(db,{'native_copy_adapter':str(target)})
    return {'installed':True,'provider':'native-copy','base_weights_sha256':base.weights_sha256,
            'adapter_weights_sha256':value['weights_sha256'],'inference_verified':False,
            'automatic_default_change':False,'capability':'experimental learned source-byte copy; general coding unverified'}
