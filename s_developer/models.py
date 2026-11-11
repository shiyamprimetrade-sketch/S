"""Explicit local native model installation and host-only configuration."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


def config_path(db):return Path(db).resolve().parent/'s-developer-host.json'


def load_config(db):
    path=config_path(db)
    value=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    if not isinstance(value,dict) or set(value)-{'native_checkpoint','ollama_model','images'}:
        raise ValueError('Unknown developer host configuration fields')
    if 'images' in value and not isinstance(value['images'],dict):raise ValueError('Images must be a mapping')
    return value


def save_config(db,updates):
    config=load_config(db);config.update(updates)
    target=config_path(db);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.json.new')
    temporary.write_text(json.dumps(config,indent=2),encoding='utf-8');temporary.replace(target)
    return config


def validate_native(path):
    from s_brain.checkpoint import load_manifest
    path=Path(path)
    if (path/'model.safetensors').stat().st_size>128000000:raise ValueError('Native checkpoint exceeds 128 MB installation budget')
    manifest=load_manifest(path)
    cfg=manifest.model_config
    if manifest.step<1 or cfg.get('vocab_size')!=256:raise ValueError('A trained S byte-model checkpoint is required')
    limits={'dim':512,'layers':8,'context':4096,'heads':16,'kv_heads':16}
    if set(cfg)!={'vocab_size',*limits}:raise ValueError('Unsupported native architecture fields')
    if any(type(cfg[k]) is not int or not 1<=cfg[k]<=v for k,v in limits.items()):raise ValueError('Native architecture exceeds supported budget')
    if cfg['dim']%cfg['heads'] or cfg['heads']%cfg['kv_heads'] or (cfg['dim']//cfg['heads'])%2:
        raise ValueError('Incompatible attention dimensions')
    return manifest


def install_native(db,checkpoint):
    checkpoint=Path(checkpoint).resolve();manifest=validate_native(checkpoint)
    root=Path(db).resolve().parent/'s-native-models'
    if root.is_symlink():raise ValueError('Symlinked model installation root')
    root.mkdir(parents=True,exist_ok=True)
    target=root/manifest.weights_sha256
    if target.is_symlink():raise ValueError('Symlinked checkpoint target')
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            staged=Path(temporary)/'checkpoint';staged.mkdir()
            for name in ('manifest.json','model.safetensors'):shutil.copyfile(checkpoint/name,staged/name)
            validate_native(staged)
            staged.rename(target)
    validate_native(target)
    save_config(db,{'native_checkpoint':str(target)})
    return {'installed':True,'provider':'native','weights_sha256':manifest.weights_sha256,
            'training_steps':manifest.step,'torch_dependency_present':importlib.util.find_spec('torch') is not None,
            'inference_verified':False,'capability':'experimental; installation is not proof of coding ability'}


def install_demo(db):
    from importlib.resources import files
    return install_native(db,Path(str(files('s_developer')/'resources'/'native-demo')))


def train_demo(output,steps=128):
    """Existing S architecture, independently initialized, synthetic-only recipe experiment."""
    import torch
    from s_brain.model import Config,SBrain
    from s_brain.checkpoint import save_checkpoint
    if type(steps) is not int or not 1<=steps<=1024:raise ValueError('Demo training budget must be 1–1024 updates')
    output=Path(output)
    if output.exists():raise FileExistsError('Training output already exists; preserve the previous checkpoint')
    torch.set_num_threads(2);torch.manual_seed(20261111)
    config=Config(dim=32,layers=1,heads=4,kv_heads=2,context=256)
    model=SBrain(config);optimizer=torch.optim.AdamW(model.parameters(),lr=0.003)
    def corpus(names):
        return ('\n'.join('Request: '+n+'\n'+json.dumps({'files':{'app.py':'print('+repr(n)+')\n','test_app.py':'# requires independent tests\n'}}) for n in names)*8).encode()
    train=corpus(['Aster','Birch','Cedar','Dune','Elm','Fern','Grove','Hazel'])
    heldout=corpus(['Iris','Juniper'])
    train_ids=torch.tensor(list(train));valid_ids=torch.tensor(list(heldout))
    valid_start=torch.arange(0,min(len(valid_ids)-97,8*96),96)
    valid_offset=torch.arange(96)
    validation_x=valid_ids[valid_start[:,None]+valid_offset]
    validation_y=valid_ids[valid_start[:,None]+valid_offset+1]
    def batch(data):
        start=torch.randint(0,len(data)-97,(4,));offset=torch.arange(96)
        return data[start[:,None]+offset],data[start[:,None]+offset+1]
    def validation():
        model.eval()
        with torch.no_grad():
            return model(validation_x,validation_y)[1].item()
    before=validation();curve=[]
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    for step in range(steps):
        model.train();x,y=batch(train_ids);optimizer.zero_grad();_,loss=model(x,y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step==0 or (step+1)%32==0 or step+1==steps:curve.append({'step':step+1,'train_nll':loss.item(),'validation_nll':validation()})
    after=validation();delta=sum((v.detach()-initial[k]).square().sum().item() for k,v in model.named_parameters())**0.5
    provenance={'independently_initialized':True,'data':'self-authored synthetic fixture; no user conversations or external teacher',
                'experiment':'small generative code-format diagnostic; not a production coder'}
    manifest=save_checkpoint(output,model,config,step=steps,train_tokens=steps*4*96,train_loss=loss.item(),validation_loss=after,
                             seed=20261111,dataset_fingerprint=hashlib.sha256(train+heldout).hexdigest(),extras=provenance)
    # Try actual neural generation and record failure honestly; never count a template as neural success.
    from .providers import NativeProvider
    spec={'name':'Kepler','answers':{'purpose':'Print a greeting'}}
    diagnostic={'passed':False}
    try:
        generated=NativeProvider(output).generate(spec,'python-cli')
        diagnostic={'passed':True,'meaning':'valid source-bundle format only; execution and usefulness unverified','files':generated}
    except Exception as error:diagnostic['error']=type(error).__name__+': '+str(error)
    report={'parameters':sum(p.numel() for p in model.parameters()),'steps':steps,'initial_validation_nll':before,
            'final_validation_nll':after,'parameter_update_l2':delta,'weights_sha256':manifest.weights_sha256,
            'provenance':provenance,'learning_curve':curve,'neural_codegen_format_diagnostic':diagnostic}
    (output/'training-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
