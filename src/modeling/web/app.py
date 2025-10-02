from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, Optional, Mapping

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from greeble.adapters.fastapi import template_response

from modeling.models.first_model import run_simulation
from modeling.models.pydantic_models import Parameters
import matplotlib.pyplot as plt

app = FastAPI(title="Token Modeling UI")

# Mount static (will host greeble CSS and any custom assets)
app.mount("/static", StaticFiles(directory=str(__file__).rsplit("/", 1)[0] + "/static"), name="static")

# Jinja templates
TEMPLATES_DIR = str(__file__).rsplit("/", 1)[0] + "/templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _plot_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _run_and_plot(params: Dict[str, Any]) -> str:
    df = run_simulation(params)
    # Dark theme for chart to match UI
    bg = "#0c0c14"
    fg = "#e8e8ea"
    accent = "#6aa1ff"
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=bg)
    ax.set_facecolor(bg)
    ax.plot(df["day"], df["price"], label="Price", color=accent, linewidth=1.8)
    ax.set_title("Token Price (AMM implied)", color=fg)
    ax.set_xlabel("Day", color=fg)
    ax.set_ylabel("Price", color=fg)
    ax.tick_params(colors=fg, labelcolor=fg)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.12))
    ax.grid(True, color=(1, 1, 1, 0.15), linewidth=0.8)
    leg = ax.legend(loc="best")
    if leg:
        leg.get_frame().set_facecolor(bg)
        leg.get_frame().set_edgecolor((1, 1, 1, 0.1))
        for text in leg.get_texts():
            text.set_color(fg)
    data_uri = _plot_to_data_uri(fig)
    plt.close(fig)
    return data_uri


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # Full page render. The chart region loads via HTMX from /chart for faster refreshes.
    context = {
        "request": request,
        "page_title": "Token Modeling",
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/chart")
async def chart(request: Request) -> HTMLResponse:
    # Use default parameters
    params = Parameters().model_dump()
    chart_uri = _run_and_plot(params)
    return template_response(
        templates,
        "partials/chart.html",
        context={"chart_data_uri": chart_uri},
        request=request,
        partial_template="partials/chart.html",
    )


@app.get("/modal/vesting")
async def vesting_modal(request: Request) -> HTMLResponse:
    # Render vesting modal with three rows by default
    return template_response(
        templates,
        "partials/vesting_modal.html",
        context={},
        request=request,
        partial_template="partials/vesting_modal.html",
    )


@app.post("/modal/vesting/submit")
async def vesting_modal_submit(request: Request) -> HTMLResponse:
    form = await request.form()
    # Collect rows by suffix indexing (0..n). Expect fields like name_0, total_tokens_0, ...
    rows = []
    i = 0
    while True:
        name = form.get(f"name_{i}")
        if name is None:
            break
        try:
            row = {
                "name": name,
                "total_tokens": float(form.get(f"total_tokens_{i}") or 0),
                "start_day": int(form.get(f"start_day_{i}") or 0),
                "cliff_days": int(form.get(f"cliff_days_{i}") or 0),
                "vest_days": int(form.get(f"vest_days_{i}") or 0),
            }
        except Exception:
            row = None
        if row:
            rows.append(row)
        i += 1

    json_text = json.dumps(rows)
    # Build a brief summary
    summary = ", ".join([f"{r['name']}({int(r['total_tokens'])})" for r in rows]) or "No schedules"

    # Return out-of-band updates to set the hidden textarea and summary, and close modal
    # modal-root will be cleared by swapping this response into it.
    content = (
        f'<div></div>'
        f'\n<textarea id="vesting_schedules" name="vesting_schedules" rows="2" style="display:none;" '
        f'hx-swap-oob="outerHTML">{json_text}</textarea>'
        f'\n<div id="vesting-summary" class="greeble-caption" style="opacity:.8" hx-swap-oob="outerHTML">{summary}</div>'
    )
    return HTMLResponse(content)

@app.get("/modal/close")
async def modal_close() -> HTMLResponse:
    # Clear modal content
    return HTMLResponse("")

@app.post("/simulate")
async def simulate(request: Request) -> HTMLResponse:
    # Gather all form fields and coerce into Parameters where possible
    form = await request.form()
    overrides: Dict[str, Any] = {}

    # Base defaults from model
    base = Parameters().model_dump()

    def coerce(name: str, value: str) -> Any:
        if value == "" or value is None:
            return None
        # Special-case vesting_schedules as JSON list
        if name == "vesting_schedules":
            try:
                data = json.loads(value)
                return data
            except Exception:
                return None
        t = type(base.get(name))
        try:
            if t is int:
                return int(value)
            if t is float:
                return float(value)
            if t is bool:
                return value.lower() in {"1", "true", "on", "yes"}
            return value
        except Exception:
            return None

    for k, v in form.items():
        if k in base:
            coerced = coerce(k, v)  # type: ignore[arg-type]
            if coerced is not None:
                overrides[k] = coerced

    # Validate with Pydantic
    p = Parameters(**{**base, **overrides})
    chart_uri = _run_and_plot(p.model_dump())

    return template_response(
        templates,
        "partials/chart.html",
        context={"chart_data_uri": chart_uri},
        request=request,
        partial_template="partials/chart.html",
    )
