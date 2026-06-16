import threading

from tqdm import tqdm as _tqdm


_tqdm.set_lock(threading.RLock())

tqdm = _tqdm
