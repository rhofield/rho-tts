import torch
from tests.test_pipeline import FakeTTS



class TestReproducibility:

    def test_repeated_job_survives_intervening_retries(self):
        tts = FakeTTS()
        tts._generate_audio = lambda text: torch.rand(16000) * .5
        first = tts.generate('Repeat me', job_id='repeat', job_seed=123)
        tts.max_iterations = 3
        tts._validate_accent_drift = lambda *a: (.9, False)
        tts.generate('Retry me', job_id='intervening', job_seed=123)
        tts.max_iterations = 1
        repeated = tts.generate('Repeat me', job_id='repeat', job_seed=123)
        assert torch.equal(first.audio, repeated.audio)
        assert tts.seed == 42
        manifest = first.acceptance['manifest']
        assert manifest == repeated.acceptance['manifest']
        assert manifest['job']['seed'] == 123
        assert first.acceptance['segments'][0]['attempts'][0]['seed'] == repeated.acceptance['segments'][0]['attempts'][0]['seed']


    def test_worker_resolves_manifest_and_replays_job(self, tmp_path):
        import io
        import json
        import soundfile as sf
        from rho_tts.isolation.worker import Worker
        from tests.test_isolation.test_proxy import TestProviderProxy
        tts = FakeTTS()
        tts._generate_audio = lambda text: torch.rand(16000) * .5
        tts._save_wav = lambda path, audio, rate: sf.write(path, audio.cpu().numpy().T, rate)
        output = io.StringIO()
        worker = Worker(protocol_out=output)
        worker._tts = tts
        proxy, transport = TestProviderProxy()._make_proxy([{'type': 'ready', 'sample_rate': 16000}])
        def send(kind, **kwargs):
            output.seek(0)
            output.truncate()
            if kind == 'manifest':
                worker._handle_manifest(kwargs)
            else:
                worker._handle_generate(kwargs)
            return json.loads(output.getvalue())
        transport.send.side_effect = send
        manifest = proxy.generation_manifest('Hello', job_id='hello', job_seed=99)
        first = proxy.generate('Hello', str(tmp_path / 'first.wav'), job_id='hello', job_seed=99)
        again = proxy.generate('Hello', str(tmp_path / 'again.wav'), job_id='hello', job_seed=99)
        assert first.acceptance['manifest']['worker'] == manifest['worker']
        assert first.acceptance['manifest']['caller'] == manifest['caller']
        assert first.acceptance['manifest']['job'] == manifest['job']
        assert (tmp_path / 'first.wav').read_bytes() == (tmp_path / 'again.wav').read_bytes()
        proxy.close()


    def test_manifest_changes_with_all_relevant_inputs(self, tmp_path):
        from rho_tts import ContinuityConfig
        from rho_tts.reproducibility import fingerprint
        reference = tmp_path / 'reference.wav'
        reference.write_bytes(b'original')
        classifier = tmp_path / 'classifier.pkl'
        classifier.write_bytes(b'classifier-a')
        tts = FakeTTS()
        tts.reference_audio_path = str(reference)
        tts.reference_text = 'Reference words'
        tts.drift_model_path = str(classifier)
        previous = None
        for change in (
            lambda: None,
            lambda: reference.write_bytes(b'new reference'),
            lambda: classifier.write_bytes(b'classifier-b'),
            lambda: setattr(tts, 'reference_text', 'New transcript'),
            lambda: setattr(tts, 'accent_drift_threshold', .5),
            lambda: setattr(tts, 'force_sentence_split', False),
            lambda: setattr(tts, 'fade_duration_sec', .3),
            lambda: setattr(tts, 'cuda_graph_depth', True),
        ):
            change()
            current = fingerprint(tts.generation_manifest('Hello'))
            assert current != previous
            previous = current
        baseline = fingerprint(tts.generation_manifest('Hello'))
        assert fingerprint(tts.generation_manifest('Hello', speed=1.2)) != baseline
        assert fingerprint(tts.generation_manifest('Hello', continuity=ContinuityConfig(previous_audio=str(reference)))) != baseline


    def test_generation_restores_host_random_streams(self):
        import random
        import numpy as np
        tts = FakeTTS()
        tts._generate_audio = lambda text: torch.rand(16000) * .5
        random.seed(15)
        np.random.seed(15)
        torch.manual_seed(15)
        expected = random.random(), np.random.random(), torch.rand(1)
        random.seed(15)
        np.random.seed(15)
        torch.manual_seed(15)
        tts.generate('Hello', job_id='hello', job_seed=99)
        assert random.random() == expected[0]
        assert np.random.random() == expected[1]
        assert torch.equal(torch.rand(1), expected[2])

    def test_repeated_job_ignores_ambient_numpy_and_python_state(self):
        import numpy as np
        import random
        tts = FakeTTS()
        tts._generate_audio = lambda text: torch.from_numpy(np.random.random(16000).astype('float32')) * random.random()
        first = tts.generate('Hello', job_id='hello', job_seed=99)
        np.random.seed(892)
        random.seed(348)
        repeated = tts.generate('Hello', job_id='hello', job_seed=99)
        assert torch.equal(first.audio, repeated.audio)
