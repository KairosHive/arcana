# hub.py -- loading a model from disk first, and from the network with patience
#
# Every `from_pretrained` asks the model hub whether its files have changed,
# even for a model that is already on disk. With huggingface-hub 0.x a dropped
# connection during that question was retried and then answered from the cache.
# With 1.x (which transformers 5 requires) the same drop ends the call: four
# operations stopped that way in one session on this machine, three of them on
# models that were already downloaded.
#
# So a model is loaded in two steps. First from disk alone, which asks the
# network nothing and is how nearly every load ends. Only when the files are
# not all there is the hub contacted, and then a dropped connection is tried
# again a few times before it is reported.
#
#     model = hub.pretrained(CLIPModel.from_pretrained, model_id)
#     path = hub.pretrained(hf_hub_download, repo, filename)
#
# It works with anything that takes `local_files_only`: transformers and
# diffusers loaders, pipeline.load_ip_adapter, hf_hub_download and
# snapshot_download.

from __future__ import annotations

import time

RETRIES = 4
WAIT = (1.0, 3.0, 8.0)

# What a network that drops looks like, by name, so neither httpx nor requests
# has to be imported here and both generations of the hub library are covered.
_TRANSIENT = ("RemoteProtocolError", "ConnectError", "ConnectionError", "ReadTimeout", "ConnectTimeout",
              "ReadError", "ProtocolError", "ChunkedEncodingError", "LocalEntryNotFoundError", "Timeout",
              "ConnectionResetError", "IncompleteRead")


def transient(error: BaseException) -> bool:
    """Whether an error is the network failing, here or as the cause of what was raised."""
    seen = 0
    while error is not None and seen < 6:
        if type(error).__name__ in _TRANSIENT or isinstance(error, (ConnectionError, TimeoutError)):
            return True
        text = str(error)
        if "Server disconnected" in text or "forcibly closed" in text or "Connection aborted" in text:
            return True
        error = error.__cause__ or error.__context__
        seen += 1
    return False


def with_retries(call, *, retries: int = RETRIES, wait=WAIT, sleep=None):
    """Run `call()`, trying again when the network drops. Any other error is raised at once."""
    sleep = sleep or time.sleep
    for attempt in range(retries):
        try:
            return call()
        except Exception as e:                              # noqa: BLE001
            if attempt == retries - 1 or not transient(e):
                raise
            pause = wait[min(attempt, len(wait) - 1)]
            print(f"[hub] the connection dropped ({type(e).__name__}); trying again in {pause:.0f} s "
                  f"({attempt + 2} of {retries}).", flush=True)
            sleep(pause)


def pretrained(loader, *args, **kwargs):
    """
    `loader(*args, **kwargs)`, from disk when the model is there, else from the hub with retries.

    A caller that passes local_files_only itself is left alone.
    """
    if "local_files_only" not in kwargs:
        try:
            return loader(*args, local_files_only=True, **kwargs)
        except Exception:                                   # noqa: BLE001
            pass            # not on disk, or not whole: ask the hub
    return with_retries(lambda: loader(*args, **kwargs))
