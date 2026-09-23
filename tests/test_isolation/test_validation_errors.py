"""Validation failures must survive the real worker/proxy protocol boundary."""

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rho_tts import ValidationError
from rho_tts.isolation.proxy import ProviderProxy
from rho_tts.isolation.worker import Worker


@pytest.mark.parametrize('mode', ['file', 'memory', 'batch', 'async', 'stream'])
def test_validation_error_round_trip(mode, tmp_path):
    output = io.StringIO()
    worker = Worker(protocol_out=output)
    worker._tts = Mock()
    message = 'Segment 1 failed speech validation after 1 attempts; audio was not accepted.'
    worker._tts.generate.side_effect = ValidationError(message)
    worker._tts.stream.side_effect = ValidationError(message)
    transport = Mock()
    paths = []

    def dispatch(kind, **payload):
        if kind == 'init':
            return {'type': 'ready', 'sample_rate': 24000}
        output.seek(0)
        output.truncate()
        paths.append(payload.get('temp_dir') or payload.get('output_path') or payload['output_base_path'])
        handler = worker._handle_stream if kind == 'stream' else worker._handle_generate
        handler(payload)
        return json.loads(output.getvalue())

    transport.send.side_effect = dispatch
    transport.send_nowait.side_effect = lambda kind, **payload: setattr(
        transport.receive, 'return_value', dispatch(kind, **payload)
    )
    with patch('rho_tts.isolation.proxy.VenvManager') as manager, \
         patch('rho_tts.isolation.proxy.WorkerProcess', return_value=transport):
        manager.return_value.ensure_venv.return_value = '/unused/python'
        with ProviderProxy('breeze') as proxy:
            with pytest.raises(ValidationError) as caught:
                if mode == 'stream':
                    list(proxy.stream('Hello'))
                elif mode == 'async':
                    asyncio.run(proxy.async_generate('Hello'))
                else:
                    texts = ['Hello', 'World'] if mode == 'batch' else 'Hello'
                    destination = None if mode == 'memory' else str(tmp_path / 'out.wav')
                    proxy.generate(texts, destination)
            assert str(caught.value) == message
    assert paths
    assert all(not Path(path).exists() for path in paths)
    if mode in ('memory', 'async'):
        assert not Path(paths[0]).parent.exists()


@pytest.mark.parametrize('error_type', [None, 'UnknownError'])
def test_unrecognized_worker_errors_keep_runtime_error(error_type):
    response = {'type': 'error', 'message': 'ordinary failure'}
    if error_type:
        response['error_type'] = error_type
    transport = Mock()
    transport.send.side_effect = [{'type': 'ready', 'sample_rate': 24000}, response]
    with patch('rho_tts.isolation.proxy.VenvManager'), \
         patch('rho_tts.isolation.proxy.WorkerProcess', return_value=transport):
        with ProviderProxy('breeze') as proxy:
            with pytest.raises(RuntimeError, match='ordinary failure'):
                proxy.generate('Hello')
