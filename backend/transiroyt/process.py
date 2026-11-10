from __future__ import annotations

import asyncio
import os
import signal


class ProcessError(RuntimeError):
    def __init__(self, returncode: int, message: str):
        super().__init__(message)
        self.returncode = returncode


async def run_process(*args: str, timeout: float = 120, cwd=None, input: bytes | None = None, process_group=False):
    process = await asyncio.create_subprocess_exec(
        *map(str, args), cwd=cwd, start_new_session=process_group,
        stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(input), timeout)
    except BaseException:
        if process.returncode is None:
            if process_group:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise ProcessError(process.returncode, stderr.decode(errors="replace")[-1800:] or f"{args[0]} failed")
    return stdout
