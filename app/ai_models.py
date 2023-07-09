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
        sanitized_hashtags = set((str(s).replace('-', '_').strip() for s in hashtags))
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
        emojis = re.sub(r'[\w\s\d,\.:�🗄▫]', '', s)
        emojis = emojis if emojis != '' else '🙂'  # use default emoji if it's empty
        return ''.join(take(n, cycle(emojis)))  # reuse emojis until size is n in case we didn't generate enough of them
    
    msg = "Generate fitting emojis for this message, but return only the emojis without any text or punctuation: " + msg

    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
        "temperature": 0.95,
        "max_output_tokens": n * 10,  # a token is about 4 characters, but we multiply by 10 to hopefully have enough different emojis
        "top_p": 0.99,
        "top_k": 40,
    }

    response = model.predict(msg, **parameters)

    return sanitize_emojis(response.text)


def unique_emojis(msg: str, n: int = 10):
    unique_emojis = set(emojis(msg, n))

    # add random emojis until we have enough unique emojis, in case we didn't have enough before
    i = 0
    while len(unique_emojis) < n + 1:
        try:
            unique_emojis.add(random_emojis[randint(0, len(random_emojis))])
            i += 1
        except IndexError:
            pass  # this happens sometimes, i would guess because some emojis are considered as multiple characters by len, but not by the indexing

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

