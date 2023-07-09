from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from vertexai.preview.language_models import ChatModel, InputOutputTextPair, TextGenerationModel
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import vertexai
import json
from fastapi.staticfiles import StaticFiles
from typing import Annotated
import re
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

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
    """
    Endpoint to handle chat.
    Receives a message from the user, processes it, and returns a response from the model.
    """
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.8,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40,
    }
    chat = chat_model.start_chat(  # Initialize the chat with model
        # chat context and examples go here
    )
    # Send the human message to the model and get a response
    response = chat.send_message(msg, **parameters)
    # Return the model's response
    return {"response": response.text}


@app.post("/summarize")
async def handle_summarize(msg: Annotated[str, Form()]):
    """
    Endpoint to generate hashtags for the given message.
    Receives a message from the user, processes it, and returns a response from the model.
    """
    return {"response": summarize(msg)}


def summarize(msg: str):    
    msg = """Please write everything in an active voice, i.e. instead of writing "the author", write "I"! Please use enthusiastic and interesting language!
    
Provide a summary with about three sentences for the following article: Beyond our own products, we think it\'s important to make it easy, safe and scalable for others to benefit from these advances by building on top of our best models. Next month, we\'ll start onboarding individual developers, creators and enterprises so they can try our Generative Language API, initially powered by LaMDA with a range of models to follow. Over time, we intend to create a suite of tools and APIs that will make it easy for others to build more innovative applications with AI. Having the necessary compute power to build reliable and trustworthy AI systems is also crucial to startups, and we are excited to help scale these efforts through our Google Cloud partnerships with Cohere, C3.ai and Anthropic, which was just announced last week. Stay tuned for more developer details soon.
Summary: Google is making its AI technology more accessible to developers, creators, and enterprises. Next month, Google will start onboarding developers to try its Generative Language API, which will initially be powered by LaMDA. Over time, Google intends to create a suite of tools and APIs that will make it easy for others to build more innovative applications with AI. Google is also excited to help scale these efforts through its Google Cloud partnerships with Cohere, C3.ai, and Anthropic.

Provide a summary with about three sentences for the following article: The benefits of electricPromptData kitchens go beyond climate impact, starting with speed. The first time I ever cooked on induction (electric) equipment, the biggest surprise was just how incredibly fast it is. In fact, induction boils water twice as fast as traditional gas equipment and is far more efficient — because unlike a flame, electric heat has nowhere to escape. At Bay View, our training programs help Google chefs appreciate and adjust to the new pace of induction. The speed truly opens up whole new ways of cooking.
Summary: Electric kitchens are faster, more efficient, and better for the environment than gas kitchens. Induction cooking is particularly fast, boiling water twice as fast as traditional gas equipment. This speed opens up whole new ways of cooking. Google chefs are trained to appreciate and adjust to the new pace of induction cooking at Bay View.

Provide a summary with about three sentences for the following article: We\'re also using AI to forecast floods, another extreme weather pattern exacerbated by climate change. We\'ve already helped communities to predict when floods will hit and how deep the waters will get — in 2021, we sent 115 million flood alert notifications to 23 million people over Google Search and Maps, helping save countless lives. Today, we\'re sharing that we\'re now expanding our coverage to more countries in South America (Brazil and Colombia), Sub-Saharan Africa (Burkina Faso, Cameroon, Chad, Democratic Republic of Congo, Ivory Coast, Ghana, Guinea, Malawi, Nigeria, Sierra Leone, Angola, South Sudan, Namibia, Liberia, and South Africa), and South Asia (Sri Lanka). We\'ve used an AI technique called transfer learning to make it work in areas where there\'s less data available. We\'re also announcing the global launch of Google FloodHub, a new platform that displays when and where floods may occur. We\'ll also be bringing this information to Google Search and Maps in the future to help more people to reach safety in flooding situations.
Summary: Google is using AI to forecast floods in South America, Sub-Saharan Africa, South Asia, and other parts of the world. The AI technique of transfer learning is being used to make it work in areas where there\'s less data available. Google FloodHub, a new platform that displays when and where floods may occur, has also been launched globally. This information will also be brought to Google Search and Maps in the future to help more people reach safety in flooding situations.

Provide a summary with about three sentences for the following article: In order to learn skiing, you must first be educated on the proper use of the equipment. This includes learning how to properly fit your boot on your foot, understand the different functions of the ski, and bring gloves, goggles etc. Your instructor starts you with one-footed ski drills. Stepping side-to-side, forward-and-backward, making snow angels while keeping your ski flat to the ground, and gliding with the foot not attached to a ski up for several seconds. Then you can put on both skis and get used to doing them with two skis on at once. Next, before going down the hill, you must first learn how to walk on the flat ground and up small hills through two methods, known as side stepping and herringbone. Now it\'s time to get skiing! For your first attempted run, you will use the skills you just learned on walking up the hill, to go down a small five foot vertical straight run, in which you will naturally stop on the flat ground. This makes you learn the proper athletic stance to balance and get you used to going down the hill in a safe, controlled setting. What do you need next? To be able to stop yourself. Here, your coach will teach you how to turn your skis into a wedge, also commonly referred to as a pizza, by rotating legs inward and pushing out on the heels. Once learned, you practice a gliding wedge down a small hill where you gradually come to a stop on the flat ground thanks to your wedge. Finally, you learn the necessary skill of getting up after falling, which is much easier than it looks, but once learned, a piece of cake.
Summary: Skiing is a great way to enjoy the outdoors and get some exercise. It can be a little daunting at first, but with a little practice, you\'ll be skiing like a pro in no time.

Provide a summary with about three sentences for the following article: Yellowstone National Park is an American national park located in the western United States, largely in the northwest corner of Wyoming and extending into Montana and Idaho. It was established by the 42nd U.S. Congress with the Yellowstone National Park Protection Act and signed into law by President Ulysses S. Grant on March 1, 1872. Yellowstone was the first national park in the U.S. and is also widely held to be the first national park in the world.The park is known for its wildlife and its many geothermal features, especially the Old Faithful geyser, one of its most popular. While it represents many types of biomes, the subalpine forest is the most abundant. It is part of the South Central Rockies forests ecoregion.
Summary: Yellowstone National Park is the first national park in the United States and the world. It is located in the western United States, largely in the northwest corner of Wyoming and extending into Montana and Idaho. The park is known for its wildlife and its many geothermal features, especially the Old Faithful geyser.

Provide a summary with about three sentences for the following article: """ + msg + """
Summary:"""

    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 500,
        "top_p": 0.95,
        "top_k": 40
    }
    response = model.predict(msg, **parameters)
    return response.text


@app.post("/hashtags")
async def handle_hashtags(msg: Annotated[str, Form()]):
    """
    Endpoint to generate hashtags for the given message.
    Receives a message from the user, processes it, and returns a response from the model.
    """
    return {"response": hashtags(msg)}


def hashtags(msg: str):
    
    def sanitize_hashtags(s):
        """
        Hashtags should only include hashtags, separated by spaces, and without duplicates.
        A hashtag is a # sign followed by letters, numbers and underscores - all other characters should be removed.

        For example:
        "#obsidian, #markdown, #notion, #org-mode, #data view, #tag folder, #openstreetmap, #acl, #markdown, #notion, #org-mode"
        should be turned into
        "#obsidian #markdown #notion #org-mode #data_view #tag_folder #openstreetmap #acl"
        """
        hashtags = re.findall('(#[\w -]+)', response.text)
        sanitized_hashtags = set((str(s).replace('-', '_') for s in hashtags))
        return ' '.join(sanitized_hashtags)

    msg = "Your response should only include hashtags, and try to generate as many hashtags as you can! Tokenize the hashtags of this transcript: " + msg

    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
        "temperature": 0.95,
        "max_output_tokens": 1024,
        "top_p": 0.99,
        "top_k": 40,
    }

    response = model.predict(msg, **parameters)
    sanitized_hashtags = sanitize_hashtags(response)

    return sanitized_hashtags


@app.post("/captionize")
async def handle_captionize(template: Annotated[str, Form()], transcript: Annotated[str, Form()]):

    result = template

    if "{summary}" in template:
        result = result.replace("{summary}", summarize(transcript))
    
    if "{hashtags}" in template:
        result = result.replace("{hashtags}", hashtags(transcript))
    
    return {"response": result}


@app.post("/transcribe")
async def handle_transcribe(audio_file: str = 'downloads/Hack Your Brain With Obsidianmd.webm'):
    """Transcribe an audio file."""
    client = SpeechClient()

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config={}, language_codes=["en-US"], model="chirp"
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/lablab-ai-hackathon/locations/global/recognizers/_",
        config=config,
        content=content,
    )

    response = client.recognize(request=request)

    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response
