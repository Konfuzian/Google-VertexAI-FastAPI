from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import vertexai
import json
from fastapi.staticfiles import StaticFiles
from typing import Annotated
import re
from ai_models import *
from transcribe_youtube import get_youtube_transcript

# Load the service account json file
# Update the values in the json file with your own
with open(
    "service_account.json"
) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
    service_account_info = json.load(f)

my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

# Initialize Google AI Platform with project details and credentials
aiplatform.init(
    credentials=my_credentials,
)

with open("service_account.json", encoding="utf-8") as f:
    project_json = json.load(f)
    project_id = project_json["project_id"]


# Initialize Vertex AI with project and location
vertexai.init(project=project_id, location="us-central1")

# Initialize the FastAPI application
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS for the application
origins = ["http://localhost", "http://localhost:8080", "http://localhost:3000"]
origin_regex = r"https://(.*\.)?alexsystems\.ai"
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("hashtag_samples.json", encoding="utf8") as f:
    hashtag_samples = json.load(f)


@app.get("/")
async def root():
    """Root endpoint that returns available endpoints in the application"""
    return {
        "Endpoints": {
            "chat": "/chat",
            "hashtags": "/hashtags",
        }
    }


@app.get("/docs")
async def get_documentation():
    """Endpoint to serve Swagger UI for API documentation"""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/redoc")
async def get_documentation():
    """Endpoint to serve ReDoc for API documentation"""
    return get_redoc_html(openapi_url="/openapi.json", title="redoc")


@app.post("/chat")
async def handle_chat(msg: Annotated[str, Form()]):
    return {"response": chat(msg)}


@app.post("/summary")
async def handle_summary(msg: Annotated[str, Form()]):
    return {"response": summary(msg)}


@app.post("/hashtags")
async def handle_hashtags(msg: Annotated[str, Form()]):
    return {"response": hashtags(msg)}


@app.post("/emojis")
async def handle_emojis(msg: Annotated[str, Form()]):
    return {"response": emojis(msg)}


@app.post("/summary_with_emojis")
async def handle_summary_with_emojis(msg: Annotated[str, Form()]):
    return {"response": summary_with_emojis(msg)}


def summary_with_emojis(msg: str):
    return add_emojis(summary(msg))


@app.post("/captionize")
async def handle_captionize(template: Annotated[str, Form()], transcript: Annotated[str, Form()]):
    return {"response": captionize(template=template, transcript=transcript)}


def captionize(template: str, transcript: str):

    if re.search(r"www.youtube.com/watch\?", transcript):
        transcript = get_youtube_transcript(transcript)
    
    print("transcript:", transcript)

    if len(transcript) > 50000:
        # keep the start and the beginnig for long messages
        transcript = transcript[:25000] + transcript[len(transcript) - 25000:]
    result = template

    if "{summary}" in template:
        for i in range(5):  # retry a few times, because this can still throw google.api_core.exceptions.InvalidArgument: 400 Request contains an invalid argument.
            try:
                result = result.replace("{summary}", summary(transcript))
                break
            except Exception as e:
                # keep only the first and the last third of the transcript (i think the middle is the least important usually...)
                print(e)
                transcript = transcript[:len(transcript) // 3] + transcript[2 * (len(transcript) // 3):]


    if "{summary-with-emojis}" in template:
        for i in range(5):  # retry a few times, because this can still throw google.api_core.exceptions.InvalidArgument: 400 Request contains an invalid argument.
            try:
                result = result.replace("{summary-with-emojis}", summary_with_emojis(transcript))
                break
            except Exception as e:
                # keep only the first and the last third of the transcript (i think the middle is the least important usually...)
                print(e)
                transcript = transcript[:len(transcript) // 3] + transcript[2 * (len(transcript) // 3):]
        

    if search := re.search(r"\{emojis\s*:?\s*(\d+)?\}", template):
        n = int(search.group(1)) if search.group(1) is not None else 10  # ugly hack, the else case should call emojis without n instead of setting n to 10
        result = result.replace(search.group(0), emojis(transcript, n=n))
    
    if search := re.search(r"\{unique-emojis\s*:?\s*(\d+)?\}", template):
        n = int(search.group(1)) if search.group(1) is not None else 10  # ugly hack, the else case should call unique_emojis without n instead of setting n to 10
        result = result.replace(search.group(0), unique_emojis(transcript, n=n))
    
    if "{hashtags}" in template:
        for i in range(3):
            tags = hashtags(transcript)
            if tags.strip() != '':  # retry if the created hashtags are empty
                result = result.replace("{hashtags}", hashtags(transcript))
                break
    
    print("result:", result)
    
    return result
