from vertexai.preview.language_models import ChatModel, InputOutputTextPair, TextGenerationModel
import re
from itertools import cycle, islice
from random import randint

### functions in this file use AI models to process the messages

# helper functions

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


# AI functions

def chat(msg: str):
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
    return response.text


def summary(msg: str):
    msg = """Please write everything in an active voice, i.e. instead of writing "the author", write "I"! Use enthusiastic and interesting language!
    
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


def hashtags(msg: str):
    
    def sanitize_hashtags(s: str):
        """
        Hashtags should only include hashtags, separated by spaces, and without duplicates.
        A hashtag is a # sign followed by letters, numbers and underscores - all other characters should be removed.

        For example:
        "#obsidian, #markdown, #notion, #org-mode, #data view, #tag folder, #openstreetmap, #acl, #markdown, #notion, #org-mode"
        should be turned into
        "#obsidian #markdown #notion #org-mode #data_view #tag_folder #openstreetmap #acl"
        """
        hashtags = re.findall(r'(#[\w -]+)', s)
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
    
    return sanitize_hashtags(response.text)


random_emojis = '❓🐹🚨↩️⏺🏄📠🈴🌋👽💳🔰🎓📺🏴⬜️🕕🍳🔼😻🐋🌖🕖🚅🕤⏱🔬♉️🕳🚧🚿🦂🍗🌏🐞⏪⁉️👰🚜📨🙍👅💢💥📟👩😢🍉📵🎹✒️💫😟8️⃣⛏👱💻🐽🔇🗂🗳🎽😉👛🐗🌻🈯️🏵🍿⛈⏲💤🌺💊🈹🍎👤🍸🐮🎻🐾🍩🍖⛸♦️🎭🎟💼🐂🙎🛡🛏🚍🚋🎫😨💣🙀🎉☠⛽️🕯🈂️🚟📋🔃🈸⚫️💈🗯⬅️💆🛃🛣🗡▶️👇🌇👬©️#️⃣⛅️🍫🕙📽🕠🏖💘🛠🔤📩🍛🈳☮😐🔎🏔🐈📑👚🍍🎋🈶🗾🚵⛲️🚸📫📊🐵☺️🏠🐲🚢👯🍦🚾🕡ℹ️🎴✅📀😬📏🎲🍑🍬🈷️💬🚁🌞🐦👍☁️📘🎩🆗🐫🛬📷🌘🏚📍🎣🦁🏨🔉😇🌱🍕🗻🎚💶🍰🔪🌴☢〽️🔄🍔🚘🙄🚮📔📙🙈🕦🎌®️🐕〰️🚗🎅♎️🕴🌟🐪🐯💎🏀🍈🏤📛🕰🍧😩🐒🚄🔋🍾🌤☦🕌🍯🏍😜📗🏞🏗😣➡️🚕🏬💺🎇🃏🍋🐩☣👓🏡👮👲💚😑⚱⤴️🔞🖍🕉❔🍭📥🍼🚉↕️🐊✌️⛓👷🕢📢🚓👫🐨🏳⏭🍡👜😖💭💂📁🔜😧👝🍏📡🌆🌗✨➕🆖👦😗🆎♠️🎺😺🌡🔕🔓😄😃🏮🌜🎥🕋🚊💸🏺🏹💐💪🍴😏Ⓜ️🚫⏮🦃💉👞🔫😦6️⃣⌚️🌲🌐👭🕜🅾️☘🌚🗼📇📓🐴🌷📉🐆➰1️⃣🔀3️⃣🍇🐺👼🔴👐🐛🚀📿📃🖐😷✍😕🕐🈁😸🔛🛅🌊🏘⛹🌠🍷💿🏊📐🤐⛄️🎾▪️😒🔖🏌🧀🍌🏃😹❇️💰🛐9️⃣🌝🖱👆🕘🆔👨🚛🚣📎2️⃣🏇💡🏙😯🏆*⃣➖🆕❄️🔹🈚️✝🏅🛋😳◼️📅🚹😥🐣🖲♋️🕗🚈💽🏁🐇🌈⛵️⏫😛♍️🔝🆓🉐🎤🚽🎞🙊🍓🏐😊👊😭🛂📝🈲👑📖㊗️🔨🍢🍽▫️🍊♈️✴️🚭😘🔂😠🦀🐍😓⏹📰🎄😼🚱🐰🌮🕸⭕️💔👀🏸🐘📻👣↔️🚡🎎❣🆙🕛🕷🛫🔡🎮🐧🔊🔟🐌☑️🐭🤑🚐🌪💀😱🐎🔽🚞5️⃣🌑👔👢🌵⬆️🌀🎈🎍🕝🛌7️⃣🎯◽️🚏🛁🌼🎨🎀😌😋⏬📜📲✳️😾👁⚔🍹📒🚠⛺️🕚🚻♿️📤🏑🐤🚥♏️🔶🌒⚙🚝🙆⚾️😤⛔️🗨📞🏉♥️🚆🐿🍪👎🏒🎒🎼↖️🐡🅱️⏰❌🌧🔻🕓🏕😵👗🏢😝🛍🐙🕍🍁🌾❎💑🚷🗽🔳🌄🌎🐑🎶📮🚴🌓⛪️◀️🍮🌹🗿🌃🖌❗️🤓🗜🌯🐀🌭😙💓🤒🍂◾️🐔🅿️🍤🔥📧🏩🐠🍃🛤🌫🕑🙋🌁🙃🎳🖨⚜🔲🏜🔐⬇️🗃👉✂️🚙🎐🎊🎑🐻💨☂🛳👳💄⚡️🌕🌥⚰⏸☪📸🍅📄🏭📳🌽🖇😁😅🎿⚠️✔️🆑☔️🚒📴📶🌍🔚➗🏈⛱🐃💌👻🌿😔◻️🌉🖖☄👙💝⌨⛷💗🕹0️⃣🆚⛰🔅⏏📹🏟🍻💜🎙🍘🎦🔱⚓️🗒🐄🤘🔙😶🕞🔸🚰🔒🍣🚤🐱🎪⛑🎰👥🏛💮🖕🗺💍🤗🔯♨️🍀🤔💴🔧👌🅰️❕⭐️🎁🔌🚎🛎😮♐️🍜🔏🏎💃👃📦🔗✖️🗝🔵🌂🍨💷⚽️😪🚌🔆🌅🐷☯🖊🔈🏝🈺↗️🎬👒📪🛀🙂⚗🙇😡🍞🐶🐁🌦💹🐥👺♒️💲🔮☝️👪📭🚩🔔🎏🕶⚪️💏✉️🎡🎵😆👋⌛️🌨💱🎗💖⚖😰🍺🍙🐚🙌🗄🔣🍚🍠⬛️🎆⏩🎸🕥😎🎠🚂💩🌛㊙️☹☸✊👵🔢⚒🎃♌️🎂📚🗣😿♻️📈🛄🚳♓️💟🔍🗞🔠🎱♣️🏋🕟👈🌙📕🖥🛥🈵🍥♑️🍱🌸↙️🛩🔩😴😞🏥👄😲👧👴💵🌔🕒🔭🍵📆😈🐉🐳🏦💛🛰🚦🐅👂✏️4️⃣🙏😫💠🚔🙅🖋🖼📂🏪⛩📬🀄️🔁🚃👿🐖🐐📼🏫🏂💕💁👖😚🏷👶🦄🔷💅☎️🐝😂🏧😽🐬🐟⛴🙉🌳🕊🍟🚯➿🐼📌☃💋🍐☕️🍄🌰🤖🎢🚺🚪™️🔘🔦💧🐸✋💞👟‼️⏯✡🍶🍲📣🔺👹🚼🏣🏏🗓💙🎛🌬👘🐜🗑🔑🏯💦🚲🌩👡🎖↘️🍆📯✈️🌌😀👕♊️❤️📱☀️🐓🕵⏳👏💒🕔🆘👾⛳️🌶🆒⤵️🐏🕣🛢🐢👠🤕🏰💾😍🏓🚚🚑⛎💯🉑🍝🚇🍒🚖🚶🎷👸🙁🚬↪️💇🕧⚛🕎🎧'

def emojis(msg: str, n: int = 10):
    """ Generate emojis that fit the message. n is the number of emojis that should be generated. """
    def sanitize_emojis(s: str):
        """ Remove anything that is not an emoji, and only return . """
        emojis = re.sub(r'[\w\s\d,\.:�🗄]', '', s)
        emojis = emojis if emojis != '' else '🙂'  # use default emoji if it's empty
        return ''.join(take(n, cycle(emojis)))  # reuse emojis until size is n in case we didn't generate enough of them
    
    msg = "Generate fitting emojis for this message, but return only the emojis without any text or punctuation: " + msg

    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
        "temperature": 0.95,
        "max_output_tokens": n * 4,  # a token is about 4 characters, so we multiply by 4
        "top_p": 0.90,
        "top_k": 40,
    }

    response = model.predict(msg, **parameters)

    return sanitize_emojis(response.text)


def unique_emojis(msg: str, n: int = 10):
    unique_emojis = set(emojis(msg, n))

    # add random emojis until we have enough unique emojis, in case we didn't have enough before
    i = 0
    while len(unique_emojis) < n + 1:
        unique_emojis.add(random_emojis[randint(0, len(random_emojis) + 1)])
        i += 1
        if i > 10000:
            break  # avoid infinite loops
    
    return ''.join(take(n, cycle(unique_emojis)))


def add_emojis(msg: str):
    msg = "Add fitting emojis to this message, but don't change the message itself: " + msg

    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
        "temperature": 0.95,
        "max_output_tokens": 1024,
        "top_p": 0.90,
        "top_k": 40,
    }

    response = model.predict(msg, **parameters)

    return response.text

