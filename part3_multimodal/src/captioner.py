"""Meme caption generator: free LLM providers (OpenAI-compatible APIs) → deterministic template fallback.

The LLM only WRITES the caption. The emotion comes from our classifier and is passed in; the LLM is told not to
contradict it. Providers are tried in order (only those with an API key in the environment / .env):

    groq        GROQ_API_KEY        openai/gpt-oss-120b        (free tier; scored 90/100 on Tunisian darija in a
                                                                 Nov-2026 20-model test — the best free option found)
    gemini      GEMINI_API_KEY      gemini-flash-lite-latest   (Google AI Studio free tier; Flash Lite: 90/100 darija)
    openrouter  OPENROUTER_API_KEY  openai/gpt-oss-120b:free   (free models, ~50 requests/day)

Override the model of a provider with env CAPTION_MODEL_<PROVIDER> (e.g. CAPTION_MODEL_GROQ). Any failure
(no key, network, timeout, rate limit, bad JSON) falls through to the next provider, then to templates.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

LANGS = ["darija", "english", "french"]
TIMEOUT_S = 10
UNSURE_CONF, UNSURE_MARGIN = 0.45, 0.10     # below → the caption jokes about the model's uncertainty
MAX_WORDS_PER_LINE = 14     # LLM lines longer than this are rejected (prompt asks for <= 10)

PROVIDERS = [
    # name, base_url, key env var, default model, extra request params
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-flash-lite-latest", {}),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-120b:free", {}),
]

# Provider order per language (first that answers wins). Gemini's Tounsi read more naturally than Groq's in a
# side-by-side check (2026-11-27); Groq is fast and good in English/French.
PROVIDER_ORDER = {"darija": ["gemini", "groq", "openrouter"], "english": ["groq", "gemini", "openrouter"],
                  "french": ["groq", "gemini", "openrouter"]}

EMOTION_WORDS = {
    "darija":  {"angry": "mnarvez", "happy": "farhan", "relaxed": "mrayyah", "sad": "7azin"},
    "english": {"angry": "angry", "happy": "happy", "relaxed": "relaxed", "sad": "sad"},
    "french":  {"angry": "énervé", "happy": "heureux", "relaxed": "détendu", "sad": "triste"},
}

# Typical canine body-language cues per emotion (general knowledge shown to the user and given to the LLM;
# NOT a claim about which pixels the classifier used).
BODY_CUES = {
    "angry": "teeth showing, wrinkled muzzle, ears pinned back, hard stare, tense body",
    "happy": "open relaxed mouth, tongue out, soft bright eyes, loose wiggly body",
    "relaxed": "soft or closed mouth, soft eyes, loose body, lying down or resting",
    "sad": "drooping ears, downcast eyes, head held low, slumped posture",
}

# ---------------------------------------------------------------- template bank (top, bottom)
T = {
    "darija": {
        "happy": [("ki tkoul 'barra n7arjou'", "w ena deja 9odem el bab"), ("el weekend wsel", "w ena farhan barcha"),
                  ("ki n9oulou 'chkoun yheb makla?'", "ENA ENA ENA"), ("ma fi rasi chay", "ken el vibes"),
                  ("rba7t el loto?", "la, 3tawni 9at3a jben"),
                  ("ma nejjemch nji lyoum, 3andi 3ors", "3ors chkoun? ... 3ors Ronaldo"),
                  ("chnowa ra2yek fel stage fi juillet?", "ena:"),
                  ("hani jey nrayyez b7dhek, t7eb njiblek 7aja?", "ey... kilo bordgen")],
        "angry": [("chkoun kla el kaskrout mte3i?!", "taw nwarrih chkoun ena"), ("9oltlek LA", "ma t3awedhech"),
                  ("ma tkallamnich taw", "ena mnarvez barcha"), ("el facteur rja3", "w ena mazelt ma nsitech"),
                  ("ki y9oulou 'douch'", "ya khouya LA, LA"),
                  ("ma ghir tbassem chwaya w ma tbayench", "behi n7awel... *ba3d 5 d9aye9*")],
        "relaxed": [("ya5i 3ich", "zéro stress, zéro mochkla"), ("ena w el canapé", "love story"),
                    ("el examen ghodwa?", "mouch mochkelti"), ("9a3ed mrayyah", "el donya tnajjem tostanna"),
                    ("ma 3andi ma na3mel", "w hedhi a7la 7aja"),
                    ("ena ba3d ma t3arakt m3a el 7ouma lkol 3al 7ess", "bech nrayyez chwaya"),
                    ("ennes lkol bdet t7adher fel PFE", "dhamiri: normal, normal")],
        "sad": [("9alouli 'nerja3 taw'", "w ma rja3ouch"), ("nhar el ithnin", "ya rabbi 3lech"),
                ("7atta 7ad ma 3tani 9at3a", "mel kaskrout"), ("el bol fergh", "w 9albi zeda"),
                ("tlabt nal3eb", "9alouli 'ba3d'"),
                ("ennes lkol mchat lel b7ar", "w ena 9a3ed nestanna fel bab")],
        "uncertain": [("mich fehem ken {emo} walla {other}", "el AI mte3na 7ayer zeda"),
                      ("{emo}? {other}?", "7atta ena ma na3rafch"),
                      ("wejhi ki y9oulou 'chbik?'", "{emo}... ama chwaya {other}")],
        "cat": [("hedha mouch kalb", "hedha 9attous ya khouya")],
        "not_a_dog": [("win el kalb?", "ma l9it 7atta kalb houni")],
        "unsure": [("fama kalb houni?", "ma n7lefech")],
    },
    "english": {
        "happy": [("when you say 'walk'", "and i'm already at the door"), ("weekend finally here", "best day of my life"),
                  ("when the treat bag opens", "pure happiness"), ("no thoughts", "just vibes"),
                  ("did i win the lottery?", "no, someone dropped cheese"),
                  ("what do you think about an internship in july?", "me:")],
        "angry": [("who ate my sandwich", "i will find you"), ("i said no", "don't ask again"),
                  ("the mailman came back", "this means war"), ("bath time?", "absolutely not"),
                  ("don't talk to me", "before my breakfast"),
                  ("them: just smile a little. me: ok, i'll try...", "*5 minutes later*")],
        "relaxed": [("exam tomorrow?", "not my problem"), ("me and the couch", "a love story"),
                    ("zero stress", "zero problems"), ("doing nothing", "and doing it well"),
                    ("the world can wait", "i'm resting"),
                    ("everyone: already working on their final project", "my conscience: normal, normal")],
        "sad": [("they said 'be right back'", "it's been 5 minutes"), ("monday again", "why"),
                ("nobody shared the sandwich", "not even a crumb"), ("empty bowl", "empty heart"),
                ("i asked to play", "they said 'later'")],
        "uncertain": [("{emo} or {other}?", "even the AI can't tell"), ("my face when", "i'm {emo}... or {other}"),
                      ("the model says {emo}", "confidence: vibes")],
        "cat": [("this is not a dog", "this is a cat, my friend")],
        "not_a_dog": [("where's the dog?", "i see no dog here")],
        "unsure": [("is there a dog here?", "i can't swear to it")],
    },
    "french": {
        "happy": [("quand tu dis 'promenade'", "j'étais déjà devant la porte"), ("enfin le week-end", "meilleur jour de ma vie"),
                  ("quand le paquet de friandises s'ouvre", "bonheur total"), ("aucune pensée", "que des good vibes"),
                  ("j'ai gagné au loto ?", "non, quelqu'un a fait tomber du fromage"),
                  ("ton avis sur un stage en juillet ?", "moi :")],
        "angry": [("qui a mangé mon sandwich", "je vais te trouver"), ("j'ai dit non", "ne redemande pas"),
                  ("le facteur est revenu", "c'est la guerre"), ("l'heure du bain ?", "hors de question"),
                  ("ne me parle pas", "avant mon petit-déj"),
                  ("eux : souris un peu. moi : ok, j'essaie...", "*5 minutes plus tard*")],
        "relaxed": [("examen demain ?", "pas mon problème"), ("moi et le canapé", "une histoire d'amour"),
                    ("zéro stress", "zéro problème"), ("ne rien faire", "et le faire bien"),
                    ("le monde peut attendre", "je me repose"),
                    ("tout le monde : déjà sur son PFE", "ma conscience : normal, normal")],
        "sad": [("ils ont dit 'je reviens'", "ça fait 5 minutes"), ("encore lundi", "pourquoi"),
                ("personne n'a partagé le sandwich", "pas même une miette"), ("gamelle vide", "cœur vide"),
                ("j'ai demandé à jouer", "ils ont dit 'plus tard'")],
        "uncertain": [("{emo} ou {other} ?", "même l'IA ne sait pas"), ("ma tête quand", "je suis {emo}... ou {other}"),
                      ("le modèle dit {emo}", "confiance : au feeling")],
        "cat": [("ceci n'est pas un chien", "c'est un chat, mon ami")],
        "not_a_dog": [("où est le chien ?", "je ne vois aucun chien ici")],
        "unsure": [("il y a un chien ici ?", "je n'en jurerais pas")],
    },
}
# Template mode + a user idea: several meme formats per emotion, so offline captions vary ({idea} = the user's text).
# (top, bottom, style) — style "bar" = modern caption band, "classic" = Impact text on the photo.
IDEA_TEMPLATES = {
    "english": {
        "happy": [("when someone says '{idea}'", "me: *already at the door*", "classic"), ("{idea}?", "say less, i'm in", "classic"),
                  ("POV: {idea}", "and i'm thrilled", "bar"), ("Nobody:", "Me hearing '{idea}': *zoomies*", "bar")],
        "angry": [("when someone says '{idea}'", "absolutely not", "classic"), ("{idea}?", "who approved this", "classic"),
                  ("POV: you mentioned {idea}", "now you'll pay", "bar"), ("Them: '{idea}'", "Me: *growls in disagreement*", "bar")],
        "relaxed": [("{idea}?", "i'll think about it tomorrow", "classic"), ("POV: {idea}", "and i'm unbothered", "bar"),
                    ("when someone says '{idea}'", "me: zero stress", "classic"), ("Them: '{idea}'", "Me: *keeps napping*", "bar")],
        "sad": [("when someone says '{idea}'", "and it's not for me", "classic"), ("{idea}?", "they went without me", "classic"),
                ("POV: {idea}", "and nobody invited me", "bar"), ("Me remembering {idea}", "*sad puppy eyes*", "bar")],
    },
    "french": {
        "happy": [("quand on dit '{idea}'", "moi : déjà devant la porte", "classic"), ("{idea} ?", "j'arrive, j'arrive", "classic"),
                  ("POV : {idea}", "et je suis ravi", "bar"), ("Personne :", "Moi en entendant '{idea}' : *zoomies*", "bar")],
        "angry": [("quand on dit '{idea}'", "c'est non.", "classic"), ("{idea} ?", "qui a validé ça ?", "classic"),
                  ("POV : tu as parlé de {idea}", "tu vas le regretter", "bar"), ("Eux : '{idea}'", "Moi : *grogne en désaccord*", "bar")],
        "relaxed": [("{idea} ?", "on verra demain", "classic"), ("POV : {idea}", "et je suis zen", "bar"),
                    ("quand on dit '{idea}'", "moi : zéro stress", "classic"), ("Eux : '{idea}'", "Moi : *continue la sieste*", "bar")],
        "sad": [("quand on dit '{idea}'", "et ce n'est pas pour moi", "classic"), ("{idea} ?", "ils y sont allés sans moi", "classic"),
                ("POV : {idea}", "et personne ne m'a invité", "bar"), ("Moi en repensant à {idea}", "*yeux de chien battu*", "bar")],
    },
    "darija": {
        "happy": [("ki y9oulou '{idea}'", "ena: deja 9odem el bab", "classic"), ("{idea}?", "yezzi ma t7kich, jit", "classic"),
                  ("POV: {idea}", "w ena farhan barcha", "bar"), ("7atta 7ad:", "Ena ki nesma3 '{idea}': nejri", "bar")],
        "angry": [("ki y9oulou '{idea}'", "LA. w yezzi.", "classic"), ("{idea}?", "chkoun 9arrar hedha?", "classic"),
                  ("POV: 7kit 3la {idea}", "taw tchouf", "bar"), ("Houma: '{idea}'", "Ena: mnarvez barcha", "bar")],
        "relaxed": [("{idea}?", "ghodwa nchoufou", "classic"), ("POV: {idea}", "w ena mrayyah", "bar"),
                    ("ki y9oulou '{idea}'", "ena: zéro stress", "classic"), ("Houma: '{idea}'", "Ena: 9a3ed nar9od", "bar")],
        "sad": [("ki y9oulou '{idea}'", "w mouch lia", "classic"), ("{idea}?", "mchaw bla bia", "classic"),
                ("POV: {idea}", "w 7atta 7ad ma 3ayatli", "bar"), ("Ena ki nfakker f {idea}", "*3inin 7zina*", "bar")],
    },
}

# "Reaction" structure from the owner's Tunisian references: a question, then "Me:" and the dog's face answers.
for _lang, (_q, _me) in {"english": ("what do you think about {idea}?", "Me:"),
                         "french": ("ton avis sur {idea} ?", "Moi :"),
                         "darija": ("chnowa ra2yek f {idea}?", "Ena:")}.items():
    for _rows in IDEA_TEMPLATES[_lang].values():
        _rows.append((_q, _me, "bar"))

LANG_RULES = {
    "darija": ("Tunisian Arabic (Tounsi / Derja) written in Latin script with Arabizi digits (3=ع, 7=ح, 9=ق, 5=خ). "
               "Use Tunisian words such as barcha, taw, chnowa, kifech, 3lech, mouch, yezzi, behi, mte3, chbik, ya5i, "
               "barra, 9a3ed, ya khouya. NEVER use Moroccan, Algerian or Egyptian forms such as bzaf, dyal, wach, "
               "ta3ek, rani, 3ayez, ezzay, keda, mazal. Use only very common everyday words; when unsure of a Tunisian "
               "word, use a simpler one or a French word, as Tunisians do. Short, natural, like a Tunisian meme page. "
               "Imitate the style of the examples given."),
    "english": "Natural internet-meme English, lowercase-friendly.",
    "french": "Français familier de mèmes internet (pas de traduction littérale).",
}

# Well-known meme FORMATS (text structures only — no copyrighted images). "classic" = white Impact text on the photo,
# "bar" = modern white caption band above the photo (Twitter/Instagram style). Each meme gets one format, so the jokes
# follow structures people recognise; "Another caption" draws a new one.
FORMATS = {
    "when": {"style": "classic", "how": "top: a relatable everyday situation starting with 'when' (translated); "
                                        "bottom: the dog's reaction, matching the emotion"},
    "that_face": {"style": "classic", "how": "top: 'that face when' (translated: Tounsi 'wejhi ki', French 'ma tête quand'); "
                                             "bottom: the situation that causes the emotion"},
    "nobody": {"style": "bar", "how": "top: exactly 'Nobody:' (Tounsi '7atta 7ad:', French 'Personne :'); "
                                      "bottom: 'Me: *the dog doing something absurd that matches the emotion*' "
                                      "(Tounsi 'Ena:', French 'Moi :')"},
    "pov": {"style": "bar", "how": "top: 'POV: ...' a situation seen from the viewer's eyes that explains the dog's face; "
                                   "bottom: a short twist, or empty"},
    "them_me": {"style": "bar", "how": "top: 'Them: <something a human says to the dog>' (Tounsi 'Houma:', French 'Eux :'); "
                                       "bottom: 'Me: <the dog's reply, matching the emotion>' (Tounsi 'Ena:', French 'Moi :')"},
    "me_after": {"style": "classic", "how": "top: 'Me after <an exhausting or absurd effort>' (Tounsi 'Ena ba3d ma ...', "
                                             "French 'Moi après ...'); bottom: a short ironic punchline about why, or empty"},
    # The next four come from Tunisian memes collected by the owner (2026-11-27): the photo IS the punchline.
    "reaction": {"style": "bar", "how": "top: a question asking for an opinion on a relatable situation (Tounsi "
                                        "'chnowa ra2yek f ...?', French 'ton avis sur ... ?'); bottom: exactly 'Me:' "
                                        "(Tounsi 'Ena:', French 'Moi :') and nothing else: the dog's face is the answer"},
    "everyone_vs_me": {"style": "bar", "how": "top: 'Everyone: <doing the serious, responsible thing>' (Tounsi "
                                              "'Ennes lkol:', French 'Tout le monde :'); bottom: 'My <conscience / "
                                              "stomach / brain>: <a reaction matching the emotion>'"},
    "five_min_later": {"style": "bar", "how": "top: someone asks the dog to behave or smile and the dog says it will try "
                                              "(Tounsi '... / behi n7awel...', French '... / ok, j'essaie...'); bottom: "
                                              "exactly '*5 minutes later*' (Tounsi '*ba3d 5 d9aye9*', French "
                                              "'*5 minutes plus tard*'): the face shows how it went"},
    "absurd_answer": {"style": "bar", "how": "top: a normal question or offer someone makes; bottom: a short, totally "
                                             "unexpected answer (a food, a random object) delivered with this emotion"},
    "me_also_me": {"style": "bar", "how": "top: 'Me: <a good resolution>' (Tounsi 'Ena:', French 'Moi :'); "
                                          "bottom: 'Also me: <breaking it 5 minutes later>' (Tounsi 'Ena zeda:', French 'Moi aussi :')"},
}

SYSTEM_PROMPT = """You write captions for funny dog memes. A computer-vision model has ALREADY classified the dog's emotion;
you never decide or change the emotion — you write a caption that fits it.
Rules:
- Output ONLY a JSON object: {"top": "...", "bottom": "..."}. Each line at most 10 words. No hashtags, no emojis.
- Follow the meme FORMAT you are given exactly: it is what makes it read as a real meme.
- Relatable, specific, a little absurd — like a popular meme page. Avoid generic lines ("best day ever").
- Family friendly — it is shown in a university classroom. Strictly forbidden, in every language including Tounsi:
  swear words (even mild, censored like "f***" or hinted), insults or name-calling of anyone (the user, the dog, a
  neighbour, an ex, a teacher...), "your mother" jokes, sexual innuendo, body shaming, violence, politics, religion.
  If the user's idea invites an insult, keep the idea but make the dog's reaction funny, never insulting.
- If the model is UNSURE (flag given), joke about the dog's ambiguous face / the AI hesitating between the two emotions.
- If the user gives an idea, build the joke around it.
- Language: """


# Safety net (the prompt alone is not enough: LLMs, Gemini included, sometimes slip vulgar or insulting Tounsi).
# A caption matching any pattern is discarded (→ next provider → templates); the user's own idea is checked too.
# Patterns are matched on the lower-cased, accent-stripped text AND on a copy with repeated letters collapsed
# ("zebbbi" → "zebi", "kouuss" → "kous"), so they are written with single letters. Word-boundary match.
# Arabizi glues little words to the front ("l3asba", "el3asba", "ya7mar", "w zebi"), so Tounsi patterns also match
# with one of those prefixes attached.
_TOUNSI_PREFIX = r"(?:ya|el|wel|bel|fel|w|b|l|f)?"
_TOUNSI = [
    # vulgarity
    "ko?u?s", "ze?bi?", "zebour", "zab", "nik", "nik(?:om|omk|ek|tek|ni|ou)", "n[iy]a?yek", "nayek", "mnayek", "manyouk",
    "nayk[a-z]*", "9a7?h?ba", "ka7?h?ba", "9a7b[a-z]*", "zok", "zokom[a-z]*", "zokek", "tabo?un", "sorm", "terma",
    "zamel", "zemla", "mibo?u?n", "3a?sb(?:a|et|ti|tek|tkom|ou)?", "3tay", "3atay", "khra", "5ra", "bzoula", "9lewi",
    "9lawi", "zoub",
    # insults and name-calling
    "7mar", "hmar", "bhim[a-z]*", "msa5", "wsa5", "ghbi", "mtakhlef", "mahbou?l", "ta7an", "9awed",
    "om+e?k", "yem+e?k", "yom+e?k", "bouk", "yel3an[a-z]*", "l3an", "tfo?u", "wled el 7ram", "bent el 7ram",
]
_BLOCKLIST = [_TOUNSI_PREFIX + "(?:" + "|".join(_TOUNSI) + ")"] + [
    # English
    "fuck[a-z]*", "f+u+c+k", "shit[a-z]*", "bitch[a-z]*", "dick[a-z]*", "pussy", "ass", "ashole", "asshole", "bastard[a-z]*",
    "slut[a-z]*", "whore[a-z]*", "cunt", "cock", "piss[a-z]*", "crap[a-z]*", "damn[a-z]*", "hell", "wtf", "stfu", "shut up",
    "idiot[a-z]*", "stupid", "dumb[a-z]*", "moron[a-z]*", "loser[a-z]*", "retard[a-z]*", "jerk", "sex[a-z]*", "porn[a-z]*",
    "your mom", "yo mama",
    # French
    "merde[a-z]*", "putain", "pute", "conard[a-z]*", "conas+e", "con", "conne", "salope", "salaud", "bite", "encul[a-z]*",
    "chier", "foutre", "bordel", "enfoire[a-z]*", "batard[a-z]*", "nique", "ntm", "fdp", "tg", "ta gueule",
    "ferme[- ]la", "ta mere", "ta race", "debile", "abruti[a-z]*", "cretin[a-z]*", "bou?fon", "couil+e[a-z]*", "cul", "naze",
]
_UNSAFE = re.compile(r"\b(" + "|".join(_BLOCKLIST) + r")\b")
_CENSORED = re.compile(r"[a-z]\*{2,}|[@#$%]{3,}")     # "f***", "$#@!" = a hinted swear word


def _normalise(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", (text or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("@", "a").replace("0", "o").replace("$", "s").replace("!", "i")
    # "f u c k", "f.u.c.k" -> "fuck": join runs of single letters split by spaces/dots/dashes
    return re.sub(r"\b(?:[a-z][\s._-]){2,}[a-z]\b", lambda m: re.sub(r"[\s._-]", "", m.group()), t)


def is_safe(text: str) -> bool:
    t = _normalise(text)
    collapsed = re.sub(r"([a-z])\1+", r"\1", t)
    return not (_UNSAFE.search(t) or _UNSAFE.search(collapsed) or _CENSORED.search(t))


_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")   # Arabic script
_FOREIGN_FORMAT_WORDS = {   # format words from another language = the model ignored the language rule
    "darija": re.compile(r"^\s*(eux|moi|personne|them|me|nobody|also me)\s*:", re.I),
    "english": re.compile(r"^\s*(eux|moi|personne|houma|ena|7atta 7ad)\s*:", re.I),
    "french": re.compile(r"^\s*(them|me|nobody|also me|houma|ena|7atta 7ad)\s*:", re.I),
}


def _well_formed(fmt: str, lang: str, out: dict) -> bool:
    """Reject LLM output that breaks its meme format (missing second line, wrong-language format words)."""
    if fmt not in ("pov", "me_after") and not out.get("bottom"):
        return False
    if any(len(out.get(k, "").split()) > MAX_WORDS_PER_LINE for k in ("top", "bottom")):
        return False                       # memes are short: a 20-word line reads like a paragraph, not a meme
    if any(_ARABIC_SCRIPT.search(out.get(k, "")) for k in ("top", "bottom")):
        return False                       # Latin script only (our meme fonts can't draw Arabic; Arabizi is the style)
    return not any(_FOREIGN_FORMAT_WORDS[lang].search(out.get(k, "")) for k in ("top", "bottom"))


def is_uncertain(confidence: float, margin: float) -> bool:
    return confidence < UNSURE_CONF or margin < UNSURE_MARGIN


def _template(kind: str, lang: str, emotion: str | None, runner_up: str | None, idea: str | None, rng) -> dict:
    words = EMOTION_WORDS[lang]
    if idea and kind in words:     # a dog emotion + a user idea
        top, bottom, style = rng.choice(IDEA_TEMPLATES[lang][kind])
        short = idea.strip()[:60]
        return {"top": top.replace("{idea}", short), "bottom": bottom.replace("{idea}", short), "style": style}
    top, bottom = rng.choice(T[lang][kind])
    fmt = {"emo": words.get(emotion or "", ""), "other": words.get(runner_up or "", "")}
    return {"top": top.format(**fmt), "bottom": bottom.format(**fmt), "style": "classic"}


_TYPO = str.maketrans({"\u2010": "-", "\u2011": "-", "\u00a0": " ", "\u202f": " "})   # odd hyphens/spaces meme fonts may not draw


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    top, bottom = (str(d.get(k, "")).strip().translate(_TYPO) for k in ("top", "bottom"))
    if not (top or bottom) or len(top) > 90 or len(bottom) > 90 or not is_safe(f"{top} {bottom}"):
        return None
    return {"top": top, "bottom": bottom}


_REJECTED: list[dict] = []   # rejected LLM answers of this session (inspected by the safety test)
_DEAD: set[str] = set()     # providers disabled for this session after an auth/config error (bad key, bad model)


def available_providers() -> list[str]:
    return [name for name, _, key, _, _ in PROVIDERS if os.environ.get(key) and name not in _DEAD]


def _llm(user_msg: str, lang: str, accept=None) -> dict | None:
    """Try providers in the language's order; return the first parsed caption that `accept` (if given) approves."""
    from openai import OpenAI
    order = PROVIDER_ORDER.get(lang, [p[0] for p in PROVIDERS])
    for name, base_url, key_env, model, extra in sorted(PROVIDERS, key=lambda p: order.index(p[0]) if p[0] in order else 99):
        key = os.environ.get(key_env)
        if not key or name in _DEAD:
            continue
        model = os.environ.get(f"CAPTION_MODEL_{name.upper()}", model)
        try:
            client = OpenAI(base_url=base_url, api_key=key, timeout=TIMEOUT_S, max_retries=0)
            resp = client.chat.completions.create(
                model=model, temperature=0.9, max_tokens=1200,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT + LANG_RULES[lang]},
                          {"role": "user", "content": user_msg}],
                extra_body=extra or None)
            out = _parse(resp.choices[0].message.content)
            if out and (accept is None or accept(out)):
                return {**out, "source": f"{name}:{model}"}
            print(f"[captioner] {name} answer rejected (unparsable, unsafe or breaks the format) -> next provider")
            _REJECTED.append({"provider": name, "lang": lang, "raw": (resp.choices[0].message.content or "")[:200]})
        except Exception as e:  # network, auth, rate limit, unsupported param… → next provider
            if getattr(e, "status_code", None) in (400, 401, 403, 404):    # won't fix itself: stop retrying
                _DEAD.add(name)
            print(f"[captioner] {name} failed: {type(e).__name__}: {str(e)[:120]}")
    return None


def generate_caption(kind: str, lang: str = "darija", emotion: str | None = None, probs: dict | None = None,
                     idea: str | None = None, use_llm: bool = True, seed: int | None = None) -> dict:
    """kind: an emotion label ('angry'…), or a gate status ('cat', 'not_a_dog', 'unsure').
    Returns {"top", "bottom", "source", "uncertain", "format", "style", "idea_blocked"}."""
    lang = lang if lang in LANGS else "darija"
    rng = random.Random(seed)
    idea = (idea or "").strip() or None
    idea_blocked = bool(idea) and not is_safe(idea)
    if idea_blocked:                # an insulting idea is ignored (it would steer the LLM and be printed by templates)
        idea = None
    uncertain, runner_up = False, None
    if probs:
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        emotion, runner_up = ranked[0][0], ranked[1][0]
        uncertain = is_uncertain(ranked[0][1], ranked[0][1] - ranked[1][1])
        kind = emotion

    if use_llm and kind in BODY_CUES and available_providers():
        conf = probs[emotion] if probs else None
        fmt = rng.choice(list(FORMATS))
        msg = (f"Meme FORMAT: {fmt} — {FORMATS[fmt]['how']}\n"
               + f"Predicted emotion: {emotion}" + (f" ({conf:.0%} confidence)" if conf is not None else "") + "\n"
               + (f"UNSURE: the model hesitates between {emotion} and {runner_up}.\n" if uncertain else "")
               + f"Typical body language for {emotion}: {BODY_CUES[emotion]}.\n"
               + (f"User's idea for the joke: {idea}\n" if idea else "")
               + "Vocabulary and tone examples in this language (follow the FORMAT above, not their structure; do not copy):\n"
               + "\n".join(f'- top: "{t}" / bottom: "{b}"' for t, b in rng.sample(T[lang][emotion], 3)) + "\n"
               + f"Write the meme caption in {lang}.")
        out = _llm(msg, lang, accept=lambda o: _well_formed(fmt, lang, o))
        if out:
            return {**out, "uncertain": uncertain, "format": fmt, "style": FORMATS[fmt]["style"],
                    "idea_blocked": idea_blocked}

    t_kind = "uncertain" if (uncertain and not idea) else kind
    out = _template(t_kind, lang, emotion, runner_up, idea, rng)
    return {"style": "classic", **out, "source": "template", "uncertain": uncertain, "format": "template",
            "idea_blocked": idea_blocked}


if __name__ == "__main__":
    print("providers with keys:", available_providers() or "none → templates")
    for lang in LANGS:
        print(lang, generate_caption("happy", lang, probs={"angry": .05, "happy": .8, "relaxed": .1, "sad": .05}, seed=1))
        print(lang, generate_caption("sad", lang, probs={"angry": .1, "happy": .1, "relaxed": .38, "sad": .42}, seed=2))
        print(lang, generate_caption("cat", lang, seed=3))
