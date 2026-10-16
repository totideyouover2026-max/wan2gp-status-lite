"""CPU-only audit using WanGP's actual dispatchers and synthetic handlers.

Run explicitly: python -B tests/upstream_enhancement_audit.py
Use --upstream-root PATH to audit a local WanGP checkout without network access.
Normal unittest discovery does not fetch or execute upstream code.
"""
import argparse
import ast
import concurrent.futures
import contextlib
import inspect
from pathlib import Path
import threading
import types
import urllib.request
from unittest.mock import patch

from test_enhancement_stages import helpers


PATHS = ('postprocessing/spatial_upsamplers.py', 'postprocessing/temporal_upsamplers.py', 'postprocessing/audio_processors.py')
ENTRIES = {'upscale_postprocessing', 'prepare_vae_upsampler', 'post_model_process_vae_upsampling',
           'temporal_upsample', '_dispatch_processor', 'generate_soundtrack', 'replace_voice_tracks',
           'process_audio_file', '_unit_compatible_progress'}


def audit(sources):
    scope = helpers()
    timer = scope['_StageTimingTelemetry']()
    epoch = timer.start_task('task', new_execution=True)
    owner = types.SimpleNamespace(_stage_timing=timer, _generation_context=threading.local(),
        _postprocessing_step_observer_installed=False, _latest_performance=None)
    owner._generation_context.execution = ('task', epoch)
    owner.set_global = lambda key, value: setattr(owner, key, value)
    modules, registries = {}, {}
    for path, source in sources.items():
        tree = ast.parse(source)
        module = types.ModuleType(path[:-3].replace('/', '.'))
        ns = module.__dict__
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try: value = ast.literal_eval(node.value)
                except (ValueError, TypeError): continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        ns[target.id] = value
                        if ('handler' in target.id and isinstance(value, list) and value and
                                all(isinstance(v, str) and '.' in v for v in value)):
                            registries[module.__name__] = value
        ns.update(inspect=inspect, attention_shared_state=contextlib.nullcontext,
            _activate_upsampler=lambda h: None, _activate_audio_processor=lambda h: None,
            offload_registry=types.SimpleNamespace(unload_vram=lambda *a: None, release_all=lambda *a: None),
            persistent_models=lambda: False, _handler_def=lambda h: {'name': h.name},
            runtime_parameter_kwargs=lambda *a: {}, compatible_loaded_model=lambda *a, **k: None,
            _release_upsampler_handler=lambda h: None, _release_audio_processor_handler=lambda h: None,
            _handler_name=lambda h: h.name, normalize_method=lambda m: m, _progress_compat_warnings=set())
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ENTRIES]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), path, 'exec'), ns)
        modules[module.__name__] = module

    def imports(name):
        if name not in modules: raise ImportError(name)
        return modules[name]
    with patch.object(scope['importlib'], 'import_module', side_effect=imports):
        scope['_install_postprocessing_step_observer'](owner)

    seen = []
    class Handler:
        name = ''
        def query_temporal_upsampler_def(self): return {'name': self.name}
        def check(self, kwargs):
            assert timer.snapshot()['last_stage'] == 'post', self.name
            callback = kwargs.get('progress_callback')
            if callback:
                inspect.signature(callback).bind('arbitrary', 1, 4, 'tiles')
                callback('Unrecognised handler conditioning', 1, 4, 'tiles')
                assert timer.snapshot()['progress']['unit'] == 'tiles', self.name
            seen.append(self.name)
        def load_upsampler(self, *a, **k): self.check(k)
        def upscale(self, *a, **k): self.check(k); return ('pixels', 'cache')
        def temporal_upsample(self, *a, **k): self.check(k); return ('pixels', 'previous', 48)
        def generate_soundtrack(self, *a, **k): self.check(k); return 'audio'
        def replace_voice_tracks(self, *a, **k): self.check(k); return (['audio'], [])
        def process_audio_file(self, *a, **k): self.check(k); return 'audio'
        def prepare_vae_upsampler(self, *a, **k): self.check(k); return Session()
        def post_model_process_vae_upsampling(self, *a, **k): self.check(k); return 'resized'
    class Session:
        def decode_inputs(self, value, **kwargs): return self.decode(value, **kwargs)
        def decode(self, value, **kwargs):
            Handler().check(kwargs)
            return value

    routes = 0
    spatial = modules['postprocessing.spatial_upsamplers']
    for name in registries.get(spatial.__name__, []):
        if name.endswith(('WanVaeUpsampler', 'MiniMaxH3VaeUpsampler')): continue
        handler = Handler(); handler.name = name
        timer.observe_progress('task', epoch, 'VAE Decoding', stage_id='decode')
        assert spatial.upscale_postprocessing(handler, 'pixels', 'future method') == ('pixels', 'cache')
        assert timer.snapshot()['enhancement'] is None
        routes += 1
    temporal = modules['postprocessing.temporal_upsamplers']
    for name in registries.get(temporal.__name__, []):
        handler = Handler(); handler.name = name
        temporal.find_temporal_upsampler = lambda method: handler
        assert temporal.temporal_upsample('future', 'pixels', None, 24) == ('pixels', 'previous', 48)
        routes += 1
    audio = modules['postprocessing.audio_processors']
    for operation in ('generate_soundtrack', 'replace_voice_tracks', 'process_audio_file'):
        handler = Handler(); handler.name = 'future ' + operation
        audio.find_processor = lambda method: handler
        args = ('future', []) if operation == 'replace_voice_tracks' else ('future',)
        getattr(audio, operation)(*args)
        assert timer.snapshot()['enhancement'] is None
        routes += 1
    handler = Handler(); handler.name = 'VAE upscaler'
    timer.observe_progress('task', epoch, 'Encoding Text Prompt', stage_id='encode')
    session = spatial.prepare_vae_upsampler(handler, 'future VAE')
    assert timer.snapshot()['last_stage'] == 'encode'
    timer.observe_progress('task', epoch, 'Denoising', stage_id='denoise')
    assert session.decode_inputs('pixels') == 'pixels'
    spatial.find_vae_upsampler = lambda method: handler
    assert spatial.post_model_process_vae_upsampling('pixels', 'future VAE') == 'resized'
    assert timer.snapshot()['enhancement'] is None
    routes += 3
    print(f'Actual upstream dispatcher audit: {routes} routes passed; {len(seen)} handler/load checks. Synthetic handlers; no GPU models run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path)
    parser.add_argument('--ref', default='main', help='WanGP GitHub branch or commit')
    args = parser.parse_args()
    def read(path):
        if args.upstream_root: return (args.upstream_root / path).read_text(encoding='utf-8-sig')
        url = f'https://raw.githubusercontent.com/deepbeepmeep/WanGP/{args.ref}/{path}'
        return urllib.request.urlopen(url, timeout=25).read().decode('utf-8-sig')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        sources = dict(zip(PATHS, pool.map(read, PATHS)))
    audit(sources)
