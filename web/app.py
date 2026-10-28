"""Run from the repository root: python -m uvicorn web.app:app --reload."""

import asyncio
import os
import json
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool

from .categories import CATEGORIES
from .diagnosis import MAX_TEXT_LENGTH, SAMPLES, diagnose, result
from .lessons import LESSONS
from .practice import check_answer, create_round
from .sentence_bank import load_bank
from .service_adapter import configured_service

WEB_ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    async def load_models():
        try:
            app.state.service, app.state.service_ready = await run_in_threadpool(
                configured_service, app.state.service_options)
        finally:
            app.state.model_loading = False

    loading = None
    if app.state.mode == "live":
        if app.state.background_model_load:
            # Set before the task starts, so the first page already says the models are loading.
            app.state.model_loading = True
            loading = asyncio.create_task(load_models())
        else:
            await load_models()
    else:
        app.state.service, app.state.service_ready = None, False
    app.state.sentence_bank = await run_in_threadpool(load_bank, app.state.bank_path)
    try:
        yield
    finally:
        if loading:
            # cancel() stops waiting; the worker thread keeps loading until the process exits.
            loading.cancel()
            with suppress(asyncio.CancelledError):
                await loading
        app.state.service, app.state.service_ready = None, False
        app.state.sentence_bank = {}


app = FastAPI(title="PatternLab", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.state.service, app.state.service_ready = None, False
app.state.service_options = None
app.state.background_model_load = False
app.state.model_loading = False
app.state.reviewer = False
app.state.bank_path = os.environ.get("PATTERNLAB_SENTENCE_BANK")
app.state.sentence_bank = {}
app.state.mode = os.environ.get("PATTERNLAB_MODE", "live").strip().lower()
if app.state.mode not in {"live", "sample"}:
    raise RuntimeError("PATTERNLAB_MODE must be live or sample")
# Only this dedicated asset directory is public, never the repository or corpus.
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=WEB_ROOT / "templates")


def model_readiness():
    """Report loaded capability, independently of the selected preview mode.

    model_loading separates a warm-up from a failed load; both have model_ready false.
    """
    ready = app.state.mode == "live" and app.state.service_ready
    loading = app.state.mode == "live" and app.state.model_loading
    return {"model_ready": ready, "model_loading": loading,
            "diagnosis_categories": sorted(CATEGORIES) if ready else []}


def page(request: Request, template: str, *, active: str, **context):
    """Render the shared layout and canonical category metadata."""
    return templates.TemplateResponse(
        request=request, name=template,
        context={"active": active, "categories": CATEGORIES, "mode": app.state.mode,
                 "reviewer": app.state.reviewer, **model_readiness(), **context},
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return page(request, "home.html", active="home")


@app.get("/practice", response_class=HTMLResponse)
def practice(request: Request):
    return page(request, "practice.html", active="practice")


@app.get("/practice/{category_id}", response_class=HTMLResponse)
def topic(request: Request, category_id: str):
    """One lesson template supplies all seven model-independent practice topics."""
    if category_id not in CATEGORIES:
        raise HTTPException(status_code=404)
    return page(request, "topic.html", active="practice",
                category=CATEGORIES[category_id], category_id=category_id,
                lesson=LESSONS[category_id])


class AnswerSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    answer: str = Field(min_length=1, max_length=100)

    @field_validator("answer")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("請先作答。")
        return value


@app.get("/api/practice/{category_id}/round")
def practice_round(category_id: str):
    try:
        return JSONResponse(create_round(category_id), headers={"Cache-Control": "no-store"})
    except KeyError:
        raise HTTPException(status_code=404, detail="找不到這個練習主題。") from None


@app.post("/api/practice/check")
def practice_check(submission: AnswerSubmission):
    try:
        return check_answer(submission.question_id, submission.answer)
    except KeyError:
        raise HTTPException(status_code=404, detail="找不到這一題。") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.get("/writing", response_class=HTMLResponse)
def writing(request: Request):
    samples = [{"label": sample["label"], "text": sample["text"]} for sample in SAMPLES.values()]
    return page(request, "writing.html", active="writing",
                samples=samples if app.state.mode == "sample" else [], max_text_length=MAX_TEXT_LENGTH,
                sentence_bank=app.state.sentence_bank,
                bank_count=sum(map(len, app.state.sentence_bank.values())))


class DiagnosisSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Blank sentence")
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ValueError("Invalid Unicode") from None
        return value  # Preserve every code point and all original whitespace.


def unique_fields(pairs):
    """Reject ambiguous JSON rather than silently keeping the last duplicate field."""
    data = dict(pairs)
    if len(data) != len(pairs):
        raise ValueError("Duplicate JSON field")
    return data


async def watch_disconnect(request, abandoned):
    """Flag a diagnosis whose browser left, so a queued request skips the model."""
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)
    abandoned.set()


@app.post("/api/diagnose")
async def diagnosis(request: Request):
    try:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 32768:
                raise ValueError("Request too large")
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise ValueError("Expected JSON")
        submission = DiagnosisSubmission.model_validate(json.loads(body, object_pairs_hook=unique_fields))
    except (ValueError, UnicodeError):
        return JSONResponse(result("invalid_input", None, app.state.mode, "invalid_input"),
                            status_code=422, headers={"Cache-Control": "no-store"})
    if app.state.mode == "live" and app.state.model_loading:
        # The page may predate this warm-up, so name it rather than saying the model is missing.
        return JSONResponse(result("generation_failed", submission.text, "live", "model_loading"),
                            status_code=503, headers={"Cache-Control": "no-store"})
    abandoned = threading.Event()
    watcher = asyncio.create_task(watch_disconnect(request, abandoned))
    try:
        payload = await run_in_threadpool(diagnose, submission.text, app.state.mode,
                                        app.state.service, app.state.reviewer, abandoned)
    finally:
        watcher.cancel()
    status_code = 503 if payload["status"] == "generation_failed" and app.state.mode == "live" else 200
    if payload["status"] == "invalid_input":
        status_code = 422
    return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    """Keep web availability separate from quiz and inference readiness."""
    return {"web_ready": True, "catalogue_ready": True, "practice_ready": True,
            **model_readiness(), "mode": app.state.mode, "checkpoint": "D", "reviewer": app.state.reviewer,
            "sentence_bank_ready": bool(app.state.sentence_bank),
            "sentence_bank_count": sum(map(len, app.state.sentence_bank.values()))}


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    if exc.status_code != 404:
        return HTMLResponse("Request could not be completed.", status_code=exc.status_code)
    return templates.TemplateResponse(
        request=request, name="not_found.html", status_code=404,
        context={"active": "", "categories": CATEGORIES, "mode": app.state.mode, "reviewer": app.state.reviewer},
    )
