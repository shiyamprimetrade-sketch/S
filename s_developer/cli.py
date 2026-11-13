"""S command launcher: python -m s_developer.cli develop application."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlencode
import webbrowser
from continuum.cli import DB, ROOT
from continuum.server import AppServer
from .intent import parse_intent
from .registry import catalog, detect

DEFAULT_DB = DB if (ROOT/'web'/'index.html').is_file() else Path(os.environ.get(
    'AXIOM_DB', str(Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.local'/'share')))/'S'/'continuum.sqlite3')))


def open_browser(url):
    """Use Termux's Android intent helper when available; no shell commands."""
    prefix = os.environ.get('PREFIX', '')
    helper = Path(prefix)/'bin'/'termux-open-url' if prefix else None
    if helper and helper.is_file():
        try:
            subprocess.run([str(helper), url], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=True, shell=False)
            return True
        except (OSError, subprocess.SubprocessError):
            print('Android browser could not be opened. Open the printed localhost URL manually.', flush=True)
            return False
    return webbrowser.open(url)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='S', description='S unified development foundation')
    parser.add_argument('--db', default=str(DEFAULT_DB))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('command', nargs='+', help='develop application | build me a website | toolchains')
    parser.add_argument('--detect', metavar='LANGUAGE', help='Explicitly probe a trusted installed toolchain')
    parser.add_argument('--enable-builds',action='store_true',help='Enable consented, container-only execution')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--model')
    parser.add_argument('--approve-download',action='store_true')
    parser.add_argument('--project')
    parser.add_argument('--recipe',choices=('static-site','python-cli','c-cli'),default='python-cli')
    parser.add_argument('--provider',choices=('template','native','native-copy','ollama-external'),default='template')
    parser.add_argument('--repair-provider',choices=('template','native','native-copy','ollama-external','requirements-literal'))
    parser.add_argument('--consent',action='store_true')
    parser.add_argument('--max-repairs',type=int,default=0)
    parser.add_argument('--backend',choices=('torch','stdlib'),default='stdlib')
    parser.add_argument('--prompt',default='Request: Hello\n')
    parser.add_argument('--decoding',choices=('sampled','greedy-json'),default='sampled')
    parser.add_argument('--tokens',type=int,default=16)
    parser.add_argument('--execute-tests',action='store_true',help='Run independent coding cases in prepared containers')
    args = parser.parse_args(argv)
    if args.command[:2] == ['rapid', 'build']:
        if not args.output:
            parser.error('--output is required for rapid build')
        if len(args.command) < 3:
            parser.error('Supply a request: S rapid build calculator')
        from s_brain.rapid_builder import generate
        try:
            result = generate(' '.join(args.command[2:]), args.output)
        except (ValueError, FileExistsError, OSError) as error:
            parser.exit(2, f'Rapid build failed: {error}\\n')
        print(json.dumps(result, indent=2))
        return 0
    from .models import load_config,save_config,install_demo,install_native,train_demo
    config=load_config(args.db)
    config['execution_enabled']=args.enable_builds
    if args.command==['evaluate','coding']:
        from .capability import CapabilityEvaluator
        if args.output and args.output.exists():parser.error('Evaluation output exists; preserve it and select a new output path')
        result=CapabilityEvaluator(config).evaluate(args.provider,args.consent,args.execute_tests,args.max_repairs)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));return 0 if result['suite_passed'] else 1
    if args.command==['model','configure-decoding']:
        from .models import configure_decoding
        print(json.dumps(configure_decoding(args.db,args.decoding),indent=2));return 0
    if args.command==['model','configure-runtime']:
        from .models import configure_runtime
        print(json.dumps(configure_runtime(args.db,args.backend),indent=2));return 0
    if args.command==['model','probe']:
        from .models import probe_native
        print(json.dumps(probe_native(config,args.prompt,args.tokens),indent=2));return 0
    if args.command==['model','install-demo']:
        print(json.dumps(install_demo(args.db),indent=2));return 0
    if args.command==['model','install-copy-adapter']:
        from .copy_models import install_copy_adapter
        if not args.adapter:parser.error('--adapter is required')
        print(json.dumps(install_copy_adapter(args.db,args.adapter),indent=2));return 0
    if args.command==['model','install-native']:
        if not args.checkpoint:parser.error('--checkpoint is required')
        print(json.dumps(install_native(args.db,args.checkpoint),indent=2));return 0
    if args.command==['model','train-demo']:
        if not args.output:parser.error('--output must name a new checkpoint directory')
        print(json.dumps(train_demo(args.output,args.steps),indent=2));return 0
    if args.command==['model','configure-ollama']:
        from .providers import OllamaProvider
        engine=OllamaProvider(args.model);provenance=engine.provenance()
        save_config(args.db,{'ollama_model':args.model})
        print(json.dumps(provenance,indent=2));return 0
    if args.command==['model','pull-ollama']:
        # Explicit, named official external model only; not labelled native S.
        if args.model!='qwen2.5-coder:0.5b':parser.error('Only the optional qwen2.5-coder:0.5b external starter is supported by this installer')
        if not args.approve_download:parser.error('Model download requires --approve-download')
        from .providers import local_json,OllamaProvider
        result=local_json('pull',{'model':args.model,'stream':False},timeout=300)
        if result.get('status')!='success':parser.error('Ollama model pull did not report success')
        provenance=OllamaProvider(args.model).provenance();save_config(args.db,{'ollama_model':args.model})
        print(json.dumps(provenance,indent=2));return 0
    if len(args.command)==3 and args.command[:2]==['runtime','prepare']:
        from .sandbox import prepare_image
        result=prepare_image(args.command[2],args.approve_download)
        if result['installed']:
            images=load_config(args.db).get('images',{});images[args.command[2]]=result['image'];save_config(args.db,{'images':images})
        print(json.dumps(result,indent=2));return 0
    if len(args.command)==2 and args.command[0]=='project':
        from .specifications import Specifications
        from continuum.storage import Store
        specs=Specifications(Store(args.db),config);engine=specs.engine
        if not args.project:parser.error('--project is required')
        if args.command[1]=='generate':
            artifacts=engine.artifacts(args.project)
            result=engine.generate({'project_id':args.project,'spec_revision':specs.latest(args.project)['revision'],
                'expected_artifact_revision':artifacts[0]['revision'] if artifacts else 0,'recipe':args.recipe,
                'provider':args.provider,'consent':args.consent})
        elif args.command[1]=='build':
            result=engine.build({'project_id':args.project,'artifact_revision':engine.latest(args.project)['revision'],
                'repair_provider':args.repair_provider or args.provider,'max_repairs':args.max_repairs,'consent':args.consent})
        elif args.command[1]=='status':result={'artifact':engine.latest(args.project),'runs':engine.runs(args.project)}
        else:parser.error('Use project generate, build, or status')
        print(json.dumps(result,indent=2))
        return 0 if args.command[1]!='build' or result['final_status']=='toolchain_passed' else 1
    if args.command == ['toolchains']:
        print(json.dumps(detect(args.detect) if args.detect else catalog(), indent=2))
        return 0
    if args.detect:
        parser.error('--detect is only supported with toolchains')
    try:
        intent = parse_intent(' '.join(args.command))
    except ValueError as error:
        parser.error(str(error))
    # Binding before opening the browser prevents accidentally using an unrelated
    # service on an occupied port. Never silently reuses or kills an existing app.
    try:
        app = AppServer('127.0.0.1', args.port, args.db,developer_config=config)
    except OSError as error:
        parser.exit(2, f'Cannot start S Developer: {error}. Choose another --port.\n')
    url = f'http://127.0.0.1:{app.server_port}/developer#' + urlencode({'request': intent['request']})
    print(f'S Developer: {url}', flush=True)
    print('Source generation and review enabled. Builds require host opt-in, action consent and prepared Docker images.', flush=True)
    # Hash fragment keeps the original request out of HTTP URLs/access logs.
    with app:
        if not args.no_browser:
            threading.Timer(0.2, lambda: open_browser(url)).start()
        try:
            app.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
