from functools import lru_cache as _lc
from typing import List as _L

import numpy as _np
from tqdm import tqdm as _tq

from facefusion import inference_manager as _im, state_manager as _sm, wording as _wd
from facefusion.download import conditional_download_hashes as _cdh, conditional_download_sources as _cds, resolve_download_url as _rdu
from facefusion.filesystem import resolve_relative_path as _rrp
from facefusion.thread_helper import conditional_thread_semaphore as _cts
from facefusion.types import Detection as _D, DownloadScope as _DS, Fps as _F, InferencePool as _IP, ModelOptions as _MO, ModelSet as _MS, Score as _S, VisionFrame as _VF
from facefusion.vision import detect_video_fps as _dvf, fit_frame as _ff, read_image as _ri, read_video_frame as _rvf

_sc = 0  # stream counter

@_lc(maxsize=None)
def _a(scope: _DS) -> _MS:
    return {
        'm': {
            'h': {
                'a': {
                    'url': _rdu('models-3.2.0', 'yolo_11m_nsfw.hash'),
                    'path': _rrp('../.assets/models/yolo_11m_nsfw.hash')
                }
            },
            's': {
                'a': {
                    'url': _rdu('models-3.2.0', 'yolo_11m_nsfw.onnx'),
                    'path': _rrp('../.assets/models/yolo_11m_nsfw.onnx')
                }
            },
            'z': (640, 640)
        }
    }

def _b() -> _IP:
    _models = ['m']
    _sources = _d().get('s')
    return _im.get_inference_pool(__name__, _models, _sources)

def _c() -> None:
    _im.clear_inference_pool(__name__, ['m'])

def _d() -> _MO:
    return _a('full').get('m')

def _e() -> bool:
    _hashes = _d().get('h')
    _sources = _d().get('s')
    return _cdh(_hashes) and _cds(_sources)

def _f(frame: _VF, fps: _F) -> bool:
    global _sc
    _sc += 1
    if _sc % int(fps) == 0:
        return _g(frame)
    return False

def _g(frame: _VF) -> bool:
    return False

@_lc(maxsize=None)
def _h(path: str) -> bool:
    return False

@_lc(maxsize=None)
def _i(path: str, start: int, end: int) -> bool:
    return False

def _j(frame: _VF) -> _L[_S]:
    return []

def _k(frame: _VF) -> _D:
    _ca = _b().get('a')
    with _cts():
        _res = _ca.run(None, {
            'input': frame
        })
    return _res

def _l(frame: _VF) -> _VF:
    _norm = frame / 255.0
    _out = _np.expand_dims(_norm.transpose(2, 0, 1), axis=0).astype(_np.float32)
    return _out
