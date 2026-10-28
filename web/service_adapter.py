"""One service seam and an allowlist for learner responses, with no model pipeline."""

import sys
from threading import Lock

from .categories import CATEGORIES
from .diagnosis import result

SERVICE_STATUSES = {"ok", "no_target_error", "abstained", "invalid_input", "unsupported_prediction", "generation_failed"}
service_lock = Lock()
# A second request waits for the first model call. The browser gives up after 65
# seconds, so stop waiting early enough for this request's own call to finish.
BUSY_WAIT_SECONDS = 30
# Each waiting request holds a worker thread that page and practice routes share,
# so only a few may wait; the rest are told the service is busy at once.
MAX_WAITING = 2
waiting = 0
waiting_lock = Lock()


def configured_service(options):
    """Load the shared service when configured; absent artifacts leave it unloaded."""
    if not options:
        return None, False
    try:
        from models.service import load_service
        return load_service(**options), True
    except Exception as error:
        # Do not expose paths, model errors or credentials in page responses or logs.
        # The class name alone tells the operator why live diagnosis is not connected.
        print(f"PatternLab: model service not loaded ({type(error).__name__})", file=sys.stderr)
        return None, False


def call_service(text, service, *, reviewer=False, abandoned=None):
    """Map the shared response plus validated prediction into the existing UI envelope.

    `abandoned` is an optional threading.Event set when the browser has gone; a
    request that waited for the lock then skips the model call."""
    global waiting
    unavailable = lambda: result("generation_failed", text, "live", "generation_unavailable")
    busy = lambda: result("generation_failed", text, "live", "service_busy")
    if service is None:
        return unavailable()
    with waiting_lock:
        if waiting >= MAX_WAITING:
            return busy()
        waiting += 1
    try:
        acquired = service_lock.acquire(timeout=BUSY_WAIT_SECONDS)
    finally:
        with waiting_lock:
            waiting -= 1
    if not acquired:
        return busy()
    if abandoned is not None and abandoned.is_set():
        service_lock.release()
        return busy()  # Nobody will read this answer, so free the model for the next learner.
    try:
        try:
            output = service.diagnose_with_prediction(text)
        except Exception as error:
            # The real service normally maps InputTooLong to a status itself. Also
            # handle it here if a compatible caller raises it directly.
            try:
                from models.service import InputTooLong
            except ImportError:
                return unavailable()
            if isinstance(error, InputTooLong):
                return result("unsupported_prediction", text, "live", "input_too_long")
            return unavailable()
        response, prediction = output["response"], output["prediction"]
        if response["contract_version"] != 1 or response["status"] not in SERVICE_STATUSES:
            raise ValueError("Unusable service response")
        # Requests have already passed web validation. An invalid shared-service
        # result is a model failure, not a request for the learner to fix the input.
        if response["status"] == "invalid_input":
            return result("generation_failed", text, "live", "invalid_prediction")
        payload = result(response["status"], text, "live", response["reason_code"])
        if response["status"] != "ok":
            return payload  # Never copy a failed generation or a speculative category.
        # The service owns scientific validation. These checks only protect the
        # cross-object UI mapping; no tokenizer, threshold or generation parser here.
        if (not prediction["has_error"] or prediction["category"] not in CATEGORIES
                or prediction["span"] != response["edit"]["span"]):
            raise ValueError("Missing validated prediction")
        payload.update(category=prediction["category"], location_kind=prediction["location_kind"],
                       edit=response["edit"], corrected_text=response["corrected_text"],
                       explanation_zh=response["explanation_zh"])
        if reviewer:
            payload["explanation_en"] = response["explanation_en"]
        return payload
    except (KeyError, TypeError, ValueError):
        return result("generation_failed", text, "live", "invalid_generation")
    finally:
        service_lock.release()
