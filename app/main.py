from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio

from app.config import settings
from app.services.prediction_service import PredictionService
from app.services.bootstrap import bootstrap_caches

app = FastAPI(title="Predictor Pro - Data-Driven Football Predictions", version="0.1.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

prediction_service = PredictionService()
scheduler = AsyncIOScheduler()


@app.on_event("startup")
async def on_startup():
    # Warm caches
    await bootstrap_caches(prediction_service)
    # Schedule periodic cache refresh at 06:00 UTC daily
    scheduler.add_job(lambda: asyncio.create_task(bootstrap_caches(prediction_service)), CronTrigger(hour=6, minute=0))
    scheduler.start()


@app.on_event("shutdown")
async def on_shutdown():
    if scheduler.running:
        scheduler.shutdown(wait=False)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "leagues": settings.default_leagues,
        },
    )


@app.get("/predict")
async def predict(
    home_team: str = Query(..., description="Home team name (Understat naming)"),
    away_team: str = Query(..., description="Away team name (Understat naming)"),
    league: str | None = Query(None, description="League short name (optional)"),
    season: int | None = Query(None, description="Season start year, e.g., 2023 for 2023/24"),
    match_date: str | None = Query(None, description="ISO date if future scheduling is known"),
):
    result = await prediction_service.predict_match(
        home_team=home_team,
        away_team=away_team,
        league=league,
        season=season,
        match_date=match_date,
    )
    return result


@app.get("/explain")
async def explain(
    home_team: str = Query(...),
    away_team: str = Query(...),
    league: str | None = None,
    season: int | None = None,
    match_date: str | None = None,
):
    result = await prediction_service.explain_prediction(
        home_team=home_team,
        away_team=away_team,
        league=league,
        season=season,
        match_date=match_date,
    )
    return result