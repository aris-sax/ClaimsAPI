from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.api.claims_api import router as claims_extraction_router
from app.config import settings
from app.dependencies import get_query_token
from app.pydantic_schemas.apis_response_utils import BasicAPIResponse

app = FastAPI(title=settings.PROJECT_NAME)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(
    claims_extraction_router,
    prefix="/claimsExtraction",
    tags=["Claims Extraction"],
    dependencies=[Depends(get_query_token)],
)


@app.get("/", response_model=BasicAPIResponse)
def read_root():
    return {"response": "I'm alive!"}
