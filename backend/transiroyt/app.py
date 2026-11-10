from __future__ import annotations

import asyncio
import hashlib
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api_models import StartFileDubbing
from .config import Settings
from .credentials import CredentialError
from .custom_text_api import CustomTextAPI, list_models
from .dubbing_options import FileDubbingOptions
from .errors import FileDubbingError
from .file_jobs import FileDubbingJobs, model_status
from .library import DubbingLibrary
from .logging_utils import configure_logging
from .mobile_workers import MobileWorkers
from .sarashina_translation import SarashinaSettings
from .sarashina_translation import load_settings as load_sarashina
from .sarashina_translation import save_settings as save_sarashina
from .task_routing import APIPreset, TaskRoutingConfig, load_routes, resolve_preset, save_routes


def create_app(settings: Settings | None = None, file_dubber_factory=None):
    configure_logging()
    settings = settings or Settings.from_env()
    instance_id = uuid.uuid4().hex
    data_id = hashlib.sha256(str(settings.data.resolve()).encode()).hexdigest()
    mobile = MobileWorkers(settings.data)
    file_jobs = FileDubbingJobs(settings, file_dubber_factory, mobile=mobile)
    library = DubbingLibrary(settings.data)

    @asynccontextmanager
    async def lifespan(app):
        await file_jobs.maintain_cache()
        stop_maintenance = asyncio.Event()

        async def maintain_cache():
            while not stop_maintenance.is_set():
                try:
                    await asyncio.wait_for(stop_maintenance.wait(), timeout=60)
                except TimeoutError:
                    await file_jobs.maintain_cache()

        await mobile.resume()
        maintenance = asyncio.create_task(maintain_cache())
        try:
            yield
        finally:
            stop_maintenance.set()
            await maintenance
            try:
                await file_jobs.shutdown()
            finally:
                await mobile.close()

    app = FastAPI(title="TransiroYT", version="0.5.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    async def authorize(request: Request):
        authorization = request.headers.get("authorization", "")
        if not settings.token or not secrets.compare_digest(authorization, f"Bearer {settings.token}"):
            raise HTTPException(401, "アプリとの接続を確認できません。TransiroYTを再起動してください。")

    auth = [Depends(authorize)]
    @app.get("/api/mobile", dependencies=auth)
    async def mobile_status():
        return mobile.public()

    @app.post("/api/mobile", dependencies=auth)
    async def mobile_enable(transport: Literal["wifi", "usb"] = "wifi"):
        try:
            return await mobile.enable(transport)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/mobile", dependencies=auth)
    async def mobile_disable():
        await mobile.stop()
        return mobile.public()

    @app.delete("/api/mobile/{device_id}", dependencies=auth)
    async def mobile_remove(device_id: str):
        mobile.remove(device_id)
        return mobile.public()

    @app.middleware("http")
    async def private_responses(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    async def health():
        # Identify the installation without exposing a path or any credentials.
        return {"service": "transiroyt", "version": "0.5.0",
                "instance_id": instance_id, "data_id": data_id}

    @app.get("/api/status", dependencies=auth)
    async def status():
        problems = [f"{binary}が見つかりません。" for binary in ["ffmpeg"] if not shutil.which(binary)]
        return {"ready": not problems, "problems": problems}

    @app.get("/api/file-dubbing", dependencies=auth)
    async def file_dubbing_status():
        return {"job": file_jobs.public(), "models": model_status(settings)}

    @app.get("/api/library", dependencies=auth)
    async def dubbing_library(directory: str | None = None):
        if directory:
            library.remember(Path(directory))
        return await asyncio.to_thread(library.snapshot)

    @app.get("/api/cache", dependencies=auth)
    async def cache_status():
        return await asyncio.to_thread(library.cache_status)

    @app.delete("/api/cache", dependencies=auth)
    async def clear_cache():
        if file_jobs.running or file_jobs.clearing_cache:
            raise HTTPException(409, "吹き替えの実行中やキャッシュの削除中は削除できません。")
        file_jobs.clearing_cache = True
        try:
            return await asyncio.to_thread(library.clear_cache)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, f"キャッシュを削除できませんでした: {exc}") from exc
        finally:
            file_jobs.clearing_cache = False

    @app.get("/api/task-routing", dependencies=auth)
    async def task_routing():
        return (await asyncio.to_thread(load_routes, settings.data)).public()

    @app.put("/api/task-routing", dependencies=auth)
    async def update_task_routing(body: TaskRoutingConfig):
        return (await asyncio.to_thread(save_routes, settings.data, body)).public()

    @app.post("/api/task-routing/presets/models", dependencies=auth)
    async def preset_models(body: APIPreset):
        try:
            config = await asyncio.to_thread(resolve_preset, settings.data, body)
            return {"models": await list_models(config)}
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/task-routing/presets/test", dependencies=auth)
    async def test_preset(body: APIPreset):
        started = time.monotonic()
        try:
            config = await asyncio.to_thread(resolve_preset, settings.data, body)
            if not config.model.strip():
                raise RuntimeError("接続テストにはモデルIDを指定してください。")
            async with asyncio.timeout(25):
                await CustomTextAPI(config).complete([{"text": "Reply with only OK."}], max_tokens=16)
            return {"ok": True, "message": "接続を確認しました。", "latency_ms": round((time.monotonic() - started) * 1000)}
        except TimeoutError:
            raise HTTPException(408, "接続テストがタイムアウトしました。URL・ネットワーク・モデルを確認してください。") from None
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/sarashina-translation", dependencies=auth)
    async def sarashina_config():
        return (await asyncio.to_thread(load_sarashina, settings.data)).public()

    @app.put("/api/sarashina-translation", dependencies=auth)
    async def update_sarashina_config(body: SarashinaSettings):
        return (await asyncio.to_thread(save_sarashina, settings.data, body)).public()

    @app.post("/api/file-dubbing", dependencies=auth, status_code=202)
    async def start_file_dubbing(body: StartFileDubbing):
        options = FileDubbingOptions(
            video=body.video or "", subtitles=body.subtitles or "", output=body.output,
            original_volume=body.original_volume, dub_volume=body.dub_volume,
            source_language=body.language, speakers=body.speakers, steps=body.steps,
            allow_download=body.allow_download)
        try:
            if not body.overwrite and Path(body.output).expanduser().exists():
                raise FileDubbingError("出力ファイルが既にあります。別の名前を指定してください。")
            return {"job": file_jobs.start(options, youtube_url=body.youtube_url, youtube_quality=body.youtube_quality,
                                           embedded_subtitles=body.video is not None and body.subtitles is None)}
        except FileDubbingError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/file-dubbing/{job_id}", dependencies=auth)
    async def cancel_file_dubbing(job_id: str):
        try:
            return {"job": await file_jobs.cancel(job_id)}
        except KeyError as exc:
            raise HTTPException(404, "ファイル吹き替えのジョブが見つかりません。") from exc

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        # Pydantic's default errors include raw request bodies (and API keys).
        return JSONResponse(status_code=422, content={"detail": "入力が不正です。URL・モデルID・設定値・作業の使用先を確認してください。外部APIにはHTTPSを使い、キーは専用欄へ入力してください。"})

    @app.exception_handler(CredentialError)
    async def credential_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        return JSONResponse(status_code=500, content={"detail": "処理に失敗しました。サーバーログを確認してください。"})

    return app
