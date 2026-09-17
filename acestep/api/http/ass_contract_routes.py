"""ASS backend-contract routes for ACE-Step.

ASS (Audio Slop Server) multiplexes many audio backends onto one GPU and speaks
a single job envelope to all of them. This module makes ACE-Step conform to that
envelope, replacing the bespoke ``/release_task`` + ``/query_result`` +
``/v1/audio`` surface with the shape every ASS backend shares:

    POST /v1/generate            -> {job_id, state}
    GET  /v1/jobs/{id}           -> {state, artifacts:[...]}
    GET  /v1/jobs/{id}/result/{name}  -> the bytes of one named artifact
    GET  /v1/info                -> {model, device, capabilities, ...}
    POST /park  /  POST /unpark  -> move the weights CPU<->GPU without a restart

The generation internals (job store, queue, single worker, the result builder in
``job_result_payload``) are untouched — we only re-dress the front door. Nothing
here is wrapped in the legacy ``{data, code, ...}`` envelope: ASS wants the bare
shapes, so that is what it gets.

Why a job is not "a WAV": ACE-Step hands back an audio clip *plus* the 5Hz
``audio_codes`` blueprint it planned *plus* the lyrics *plus* the metas/seed. The
artifact is the unit — see ``_artifacts_for`` — so all of that survives the trip
instead of the caller having to scrape it out of a stringly-typed ``result``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from acestep.api.http.ass_artifacts import artifacts_for, public
from acestep.api.http.release_task_request_parser import parse_release_task_request


def _artifacts_for(record: Any) -> List[Dict[str, Any]]:
    """Artifacts for a finished job record (delegates to the pure helper)."""
    return artifacts_for(getattr(record, "result", None) or {})


def _initialized_handlers(app: FastAPI) -> List[Any]:
    """Every DiT handler that has actually loaded a model (skip empty slots)."""
    handlers = []
    for attr in ("handler", "handler2", "handler3"):
        h = getattr(app.state, attr, None)
        if h is not None and getattr(h, "model", None) is not None:
            handlers.append(h)
    return handlers


def register_ass_contract_routes(
    app: FastAPI,
    *,
    store: Any,
    verify_token_from_request: Callable[[dict, Optional[str]], Optional[str]],
    request_parser_cls: Any,
    request_model_cls: Any,
    validate_audio_path: Callable[[Optional[str]], Optional[str]],
    save_upload_to_temp: Callable[..., Any],
    upload_file_type: type,
    default_dit_instruction: str,
    lm_default_temperature: float,
    lm_default_cfg_scale: float,
    lm_default_top_p: float,
    get_project_root: Callable[[], str],
    get_model_name: Callable[[str], str],
    collect_model_inventory: Callable[..., Dict[str, Any]],
) -> None:
    """Register the ASS-contract routes on ``app``."""

    # ---- submit -----------------------------------------------------------

    @app.post("/v1/generate")
    async def generate(request: Request, authorization: Optional[str] = Header(None)):
        """Submit a generation job. Returns the bare ``{job_id, state}`` envelope.

        Accepts the same JSON or multipart body the old ``/release_task`` took —
        multipart is how a source clip (``ctx_audio``/``ref_audio``) rides along
        for cover/repaint — we just parse it and hand back ASS's shape.
        """
        req, temp_files = await parse_release_task_request(
            request=request,
            authorization=authorization,
            verify_token_from_request=verify_token_from_request,
            request_parser_cls=request_parser_cls,
            request_model_cls=request_model_cls,
            validate_audio_path=validate_audio_path,
            save_upload_to_temp=save_upload_to_temp,
            upload_file_type=upload_file_type,
            default_dit_instruction=default_dit_instruction,
            lm_default_temperature=lm_default_temperature,
            lm_default_cfg_scale=lm_default_cfg_scale,
            lm_default_top_p=lm_default_top_p,
        )

        record = store.create()
        queue_ref: asyncio.Queue = app.state.job_queue
        if queue_ref.full():
            for temp_path in temp_files:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise HTTPException(status_code=429, detail="Server busy: queue is full")

        if temp_files:
            async with app.state.job_temp_files_lock:
                app.state.job_temp_files[record.job_id] = temp_files

        async with app.state.pending_lock:
            app.state.pending_ids.append(record.job_id)

        await queue_ref.put((record.job_id, req))
        return {"job_id": record.job_id, "state": "queued"}

    # ---- poll -------------------------------------------------------------

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        """Poll one job. ``artifacts`` is populated once the job succeeds."""
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

        body: Dict[str, Any] = {"job_id": job_id, "state": record.status}
        if record.status == "succeeded":
            body["artifacts"] = [public(a) for a in _artifacts_for(record)]
        else:
            body["artifacts"] = []
        if record.status == "failed" and getattr(record, "error", None):
            body["error"] = record.error
        return body

    # ---- download one artifact -------------------------------------------

    @app.get("/v1/jobs/{job_id}/result/{name}")
    async def get_artifact(job_id: str, name: str):
        """Serve one named artifact from a finished job."""
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        if record.status != "succeeded":
            raise HTTPException(
                status_code=409, detail=f"Job {job_id} is {record.status}, not succeeded"
            )

        for artifact in _artifacts_for(record):
            if artifact["name"] != name:
                continue
            kind, payload = artifact["_src"]
            if kind == "file":
                if not os.path.isfile(payload):
                    raise HTTPException(status_code=410, detail="Artifact file is gone")
                return FileResponse(payload, media_type=artifact["content_type"], filename=name)
            return Response(content=payload, media_type=artifact["content_type"])

        raise HTTPException(status_code=404, detail=f"No artifact named {name!r} on job {job_id}")

    # ---- info -------------------------------------------------------------

    @app.get("/v1/info")
    async def info():
        """Model / device / capabilities — bare, no legacy envelope."""
        inventory = collect_model_inventory(app, get_project_root, get_model_name)
        handlers = _initialized_handlers(app)
        device = getattr(handlers[0], "device", "cpu") if handlers else "cpu"
        sample_rate = getattr(handlers[0], "sample_rate", None) if handlers else None
        return {
            "service": "acestep",
            "verb": "generate",
            "model": inventory.get("default_model"),
            "lm_model": inventory.get("loaded_lm_model"),
            "device": device,
            "sample_rate": sample_rate,
            "parked": bool(getattr(app.state, "parked", False)),
            "initialized": bool(getattr(app.state, "_initialized", False)),
            # Task types the DiT can run; audio_format is the caller's choice.
            "capabilities": {
                "tasks": ["text2music", "cover", "repaint", "extract"],
                "audio_formats": ["mp3", "flac", "opus", "aac", "wav", "wav32"],
            },
        }

    # ---- park / unpark ----------------------------------------------------

    @app.post("/park")
    async def park():
        """Demote to CPU RAM: move the weights off the GPU, free the VRAM.

        Refused with 409 if a job is running — we do not yank the card out from
        under an in-flight generation (ASS's leases already prevent this, but the
        backend defends itself too). Serialized against model init/reinit via the
        shared init lock.
        """
        return await _do_park(app, store, want_parked=True)

    @app.post("/unpark")
    async def unpark():
        """Promote back to the GPU. The fast path — no container restart."""
        return await _do_park(app, store, want_parked=False)


def _running_count(store: Any) -> int:
    """How many jobs the store currently has in the ``running`` state."""
    if store is not None and hasattr(store, "get_stats"):
        try:
            return int(store.get_stats().get("running", 0))
        except Exception:
            return 0
    return 0


async def _do_park(app: FastAPI, store: Any, *, want_parked: bool) -> JSONResponse:
    import torch  # local import: ASS-side tooling can import this module GPU-free

    # Never move weights mid-generation. Single worker => at most one running.
    running = _running_count(store)
    if running > 0:
        return JSONResponse(
            status_code=409,
            content={"error": f"{running} job(s) running; refuse to move weights", "parked": bool(getattr(app.state, "parked", False))},
        )

    already = bool(getattr(app.state, "parked", False))
    if want_parked == already:
        # Idempotent no-op — asking to park something already parked is fine.
        return JSONResponse(content={"parked": already, "changed": False})

    handlers = _initialized_handlers(app)
    if not handlers:
        # Nothing loaded yet; record the intent so /health/info stays honest.
        app.state.parked = want_parked
        return JSONResponse(content={"parked": want_parked, "changed": True, "note": "no model loaded"})

    lock = getattr(app.state, "_init_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        for handler in handlers:
            target = "cpu" if want_parked else getattr(handler, "device", "cpu")
            for component_name in ("model", "vae", "text_encoder"):
                component = getattr(handler, component_name, None)
                if component is None:
                    continue
                # Reuse the handler's own battle-tested transfer path (handles
                # quantized / LoRA weights model.to() would miss).
                handler._recursive_to_device(component, target)
            # empty_cache() is mandatory on park or the VRAM never actually frees.
            if hasattr(handler, "_empty_cache"):
                handler._empty_cache()
        if not want_parked and torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        if lock is not None:
            lock.release()

    app.state.parked = want_parked
    return JSONResponse(content={"parked": want_parked, "changed": True})
