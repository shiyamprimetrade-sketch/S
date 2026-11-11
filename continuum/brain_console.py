"""Project-scoped integration of the existing S-Brain components into Continuum.

Only an operator-owned local JSON file enables features. Request payloads cannot
change host permissions, select filesystem paths, or grant model text authority.
"""
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

MODEL = 's-brain:experimental'


class BrainConsole:
    def __init__(self, store, root=None, configuration=None):
        self.store = store
        self.root = Path(root or Path(__file__).resolve().parents[1])
        path = self.root / 's-brain-host.json'
        self.config = configuration if configuration is not None else (
            json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {})
        for flag in ('enabled', 'memory_enabled', 'tools_enabled'):
            if type(self.config.get(flag, False)) is not bool:
                raise ValueError('Host feature flags must be booleans')
        self.lock = threading.RLock()
        self.model = self.manifest = self.memory = None

    def _checkpoint(self):
        value = self.config.get('checkpoint')
        if not self.config.get('enabled') or not value:
            raise PermissionError('S Native is disabled or has no configured checkpoint')
        return self.root / value

    def _load(self):
        from s_brain.checkpoint import load_manifest, load_weights
        checkpoint = self._checkpoint()
        manifest = load_manifest(checkpoint)
        if manifest.step < 1:
            raise ValueError('Untrained checkpoint cannot be served as S Native')
        with self.lock:
            if self.manifest != manifest:
                import torch
                torch.set_num_threads(2)
                self.model, self.manifest = load_weights(checkpoint, 'cpu')
        return self.model

    def status(self):
        state = {'model': MODEL, 'label': 'S Native (experimental)', 'available': False,
                 'memory_enabled': self.config.get('memory_enabled', False),
                 'tools_enabled': self.config.get('tools_enabled', False),
                 'automatic_capture': False, 'teacher': 'Optional Ollama Teacher; never a native fallback',
                 'metrics': {}, 'error': None}
        try:
            model = self._load()
            m = self.manifest
            state.update(available=True, context=model.cfg.context,
                         checkpoint_hash=m.weights_sha256,
                         metrics={'parameters': sum(p.numel() for p in model.parameters()),
                                  'training_steps': m.step, 'training_loss': m.train_loss,
                                  'validation_loss': m.validation_loss,
                                  'dataset_version': m.dataset_fingerprint,
                                  'gradient_norm': m.extras.get('gradient_norm', 'Not measured'),
                                  'weight_change': m.extras.get('weight_change', 'Not measured'),
                                  'provenance': m.extras.get('provenance', 'Unavailable'),
                                  'benchmark': m.extras.get('benchmark', 'Not measured'),
                                  'learning_curve': m.extras.get('learning_curve', 'Not measured'),
                                  'best_checkpoint_hash': m.extras.get('best_checkpoint_hash', 'Not measured'),
                                  'hardware': m.runtime, 'promotion': 'Experimental; not production-promoted'})
        except (OSError, ValueError, ImportError, RuntimeError, PermissionError, TypeError, KeyError) as error:
            state['error'] = str(error)
        return state

    def _scope(self, project_id):
        if not isinstance(project_id, str) or not any(p['id'] == project_id for p in self.store.list_projects()):
            raise ValueError('Select an existing project')
        return project_id

    def _memory(self):
        if not self.config.get('memory_enabled', False):
            raise PermissionError('Memory disabled by host')
        from s_brain.memory_store import MemoryStore
        with self.lock:
            if self.memory is None:
                self.memory = MemoryStore(self.store.path.parent / 's-brain-memory.sqlite3')
        return self.memory

    def request(self, verb, path, query, body):
        if path == '/api/brain/status' and verb == 'GET':
            return self.status()
        if verb == 'GET':
            scope = self._scope(query.get('project_id', [''])[0])
            if path == '/api/brain/memory':
                return [asdict(row) for row in self._memory().list_scope(scope)]
            if path == '/api/brain/search':
                return [dict(asdict(hit.record), score=hit.score) for hit in self._memory().retrieve(
                    scope, query.get('q', [''])[0])]
            if path == '/api/brain/audit':
                return self._memory().audit(scope)
        elif verb == 'POST':
            if not isinstance(body, dict):
                raise ValueError('JSON object required')
            if path == '/api/brain/assist':
                return self.chat(body)
            scope = self._scope(body.get('project_id'))
            if body.get('consent') is not True:
                raise PermissionError('Explicit user confirmation is required')
            memory = self._memory()
            if path == '/api/brain/remember':
                ttl = body.get('ttl_seconds', 86400)
                if type(ttl) is not int:
                    raise ValueError('TTL must be integer seconds')
                return asdict(memory.remember(scope=scope, content=body.get('content'),
                    source=body.get('source'), consent='explicit', ttl_seconds=ttl))
            if path == '/api/brain/update':
                try:
                    return asdict(memory.update(scope, body.get('id'), body.get('content'), consent='explicit'))
                except KeyError as error:
                    raise ValueError('Memory absent in selected project') from error
            if path == '/api/brain/forget':
                return {'deleted': memory.forget(scope, body.get('id'))}
        raise ValueError('Unknown S-Brain route')

    def chat(self, body):
        from s_brain.assist_runtime import AssistantEngine, AssistRequest, AssistConfiguration
        from s_brain.tools_control import ToolPermissionError
        session = self.store.get_session(body.get('session_id', ''))
        scope = session['project_id']
        if body.get('project_id', scope) != scope:
            raise PermissionError('Session belongs to a different project')
        memory_requested = body.get('memory', False)
        tools_requested = body.get('tool_intents', [])
        if tools_requested and body.get('approve_tools') is not True:
            raise PermissionError('Explicit approval required for proposed tools')
        config = AssistConfiguration(enabled=bool(self.config.get('enabled')),
            memory_enabled=self.config.get('memory_enabled', False),
            tools_enabled=self.config.get('tools_enabled', False), allowed_scope=scope)
        request = AssistRequest.parse({'question': body.get('text'), 'scope': scope,
            'memory': memory_requested, 'cite_sources': body.get('cite_sources', False),
            'tool_intents': tools_requested, 'max_new_tokens': 32, 'seed': 7}, config.limits)
        start = time.monotonic()
        with self.lock:
            model = self._load()
            engine = AssistantEngine(model, configuration=config,
                                     memory=self._memory() if memory_requested else None)
            try:
                result = engine.respond(request)
            except ToolPermissionError as error:
                raise PermissionError(str(error)) from error
            self.store.add_message(session['id'], 'user', request.question)
            message = self.store.add_message(session['id'], 'assistant',
                '[S Native experimental; reasoning capability unverified]\n' +
                (result['response'].strip() or '[The checkpoint generated no readable answer.]'),
                model=MODEL, elapsed_ms=int(1000*(time.monotonic()-start)))
            self.store.log(scope, 's-brain-assist', json.dumps(result['audit'], sort_keys=True))
        return dict(message, brain=result)
