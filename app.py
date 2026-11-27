"""DogTranslator — Gradio app. UI only: all logic lives in part3_multimodal/src/pipeline.py.

Run:  docker compose up --build   →  http://localhost:7860
"""
from __future__ import annotations

import html
import random
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "part3_multimodal" / "src"))
import captioner  # noqa: E402
import emotion  # noqa: E402
import meme_maker  # noqa: E402
import pipeline  # noqa: E402

EX = ROOT / "part3_multimodal" / "assets" / "examples"
LANG_CHOICES = [("Tounsi", "darija"), ("English", "english"), ("Français", "french")]
EMO_COLORS = {"angry": "#ED5800", "happy": "#F4B400", "relaxed": "#148F9E", "sad": "#0B2B3C"}
TEXT_EXAMPLES = ["a grumpy pug judging you", "a happy dog running on the grass", "a sad puppy in the rain",
                 "a sleepy dog on the sofa", "an angry dog showing its teeth"]

# ------------------------------------------------------------------ HTML fragments
def gate_html(res) -> str:
    g = res.get("gate")
    if g is None:   # text mode: dataset photos already passed Part 1's gate
        h = res["chosen"]
        return (f'<div class="pill pill-ok">Found in the dataset (CLIP match {h["similarity"]:.2f})</div>'
                f'<p class="fine">Dataset label: <b>{h["label"]}</b> (human label, can be noisy)</p>')
    if g.is_dog:
        extra = f'<p class="fine">{html.escape(g.note)}</p>' if g.note else ""
        return f'<div class="pill pill-ok">Dog found ({g.dog_conf:.0%} sure)</div>{extra}'
    msg = {"cat": "This looks like a cat, not a dog.", "not_a_dog": "No dog found in this photo.",
           "unsure": "I can barely see a dog here."}[g.status]
    return (f'<div class="pill pill-no">{msg}</div>'
            '<p class="fine">The emotion model only reads dogs, so it was skipped. Try a clearer dog photo.</p>')


def bars_html(emo) -> str:
    if not emo:
        return ""
    rows = []
    for lab in emotion.LABELS:
        p = emo["probs"][lab]
        top = " top" if lab == emo["emotion"] else ""
        rows.append(f'<div class="bar{top}"><span class="lab">{lab}</span>'
                    f'<span class="track"><span class="fill {lab}" style="width:{p*100:.1f}%;background:{EMO_COLORS[lab]}"></span></span>'
                    f'<span class="val">{p:.0%}</span></div>')
    unsure = ""
    if captioner.is_uncertain(emo["confidence"], emo["margin"]):
        unsure = f'<p class="fine">The model hesitates between {emo["emotion"]} and {emo["runner_up"]}.</p>'
    return f'<div class="bars">{"".join(rows)}</div>{unsure}'


def details_html(res) -> str:
    emo, cap, t = res.get("emotion"), res["caption"], res["timings"]
    parts = []
    if cap.get("idea_blocked"):
        parts.append('<p><b>Your joke idea was left out:</b> it contains a word the caption filter blocks. '
                     'Memes here stay family friendly.</p>')
    if emo:
        parts.append(f'<p><b>Typical {emo["emotion"]} body language:</b> {res["body_cues"]}.</p>')
    src = cap["source"]
    who = "built-in templates (offline)" if src == "template" else src.split(":", 1)[1] + f" via {src.split(':')[0].title()}"
    fmt = cap.get("format", "template").replace("_", " ")
    parts.append(f'<p class="fine">Meme format: {html.escape(fmt)}. Caption written by {html.escape(who)}. '
                 f'Emotion by {emo["model"] if emo else "—"}.</p>')
    ms = ", ".join(f'{k.replace("_ms", "")} {v:.0f} ms' for k, v in t.items())
    parts.append(f'<p class="fine">Timing: {ms}.</p>')
    return "".join(parts)


# ------------------------------------------------------------------ callbacks
def make_from_photo(img, idea, lang, use_llm):
    if img is None:
        raise gr.Error("Add a photo first: upload one, paste it, or pick an example below.")
    seed = random.randrange(10**6)
    res = pipeline.run_image(img, idea=idea, lang=lang, use_llm=use_llm, seed=seed)
    state = {"img": res["gate"].crop or img, "emotion": res["emotion"], "kind": res["gate"].status if not res["gate"].is_dog else None,
             "idea": idea, "timings": res["timings"], "body_cues": res.get("body_cues"), "gate": res["gate"]}
    return res["meme"], gate_html(res), bars_html(res["emotion"]), details_html(res), state


def search_photos(query):
    if not query or not query.strip():
        raise gr.Error("Describe the dog you want, for example: a grumpy pug judging you.")
    try:
        hits = pipeline.search(query, k=4)
    except FileNotFoundError:
        raise gr.Error("The photo index isn't built yet. Run part3_multimodal/src/build_index.py once.")
    return [(h["path"], f'{h["similarity"]:.2f}') for h in hits], hits


def make_from_text(query, lang, use_llm, hits, pick=0):
    if not hits:
        _, hits = search_photos(query)
    res = pipeline.run_text(query, lang=lang, use_llm=use_llm, hits=hits, pick=pick, seed=random.randrange(10**6))
    state = {"img": Image.open(res["chosen"]["path"]).convert("RGB"), "emotion": res["emotion"], "kind": None,
             "idea": query, "timings": res["timings"], "body_cues": res["body_cues"], "chosen": res["chosen"]}
    return res["meme"], gate_html(res), bars_html(res["emotion"]), details_html(res), state


def search_and_make(query, lang, use_llm):
    gallery, hits = search_photos(query)
    return (gallery, hits, *make_from_text(query, lang, use_llm, hits))


def pick_photo(query, lang, use_llm, hits, evt: gr.SelectData):
    return make_from_text(query, lang, use_llm, hits, pick=evt.index)


def new_caption(state, lang, use_llm):
    """Same photo and emotion, new joke: no models re-run except the caption."""
    if not state:
        raise gr.Error("Make a meme first.")
    emo = state["emotion"]
    if emo:
        cap = captioner.generate_caption(emo["emotion"], lang, probs=emo["probs"], idea=state["idea"],
                                         use_llm=use_llm, seed=random.randrange(10**6))
    else:
        cap = captioner.generate_caption(state["kind"], lang, seed=random.randrange(10**6))
    return meme_maker.render(state["img"], cap)


# ------------------------------------------------------------------ layout
# Two designed palettes. Gradio switches to dark mode from the browser/OS setting or its own settings panel
# (it puts a "dark" class on the page); every colour below goes through CSS variables that flip with it.
PALETTE = {"sand": "#F2E8D5", "paper": "#FFFDF8", "midnight": "#0B2B3C", "deep": "#071F2B", "panel": "#12384D",
           "ember": "#ED5800", "lagoon": "#148F9E", "mustard": "#F4B400"}

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@600;700&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
"""

CSS = """
:root { --sand:#F2E8D5; --midnight:#0B2B3C; --ember:#ED5800; --lagoon:#148F9E; --mustard:#F4B400;
  --bg: var(--sand); --surface:#FFFDF8; --ink: var(--midnight); --track: rgba(11,43,60,.12);
  --steps-bg: var(--midnight); --steps-ink: var(--sand); --meme-bg: var(--midnight); --pill-no-bg: var(--midnight); --pill-no-ink: var(--sand); }
.dark { --bg:#0B2B3C; --surface:#12384D; --ink:#F2E8D5; --track: rgba(242,232,213,.16);
  --steps-bg:#071F2B; --steps-ink:#F2E8D5; --meme-bg:#071F2B; --pill-no-bg:#F2E8D5; --pill-no-ink:#0B2B3C; }
html { background: var(--sand); } html:has(.dark) { background: #0B2B3C; }
body, gradio-app, .gradio-container, .main, .contain { background: var(--bg) !important; }
body, .gradio-container { color: var(--ink); font-family: 'Work Sans', system-ui, sans-serif; }
/* full width: the page uses the whole screen, with side padding that grows on wide monitors */
.gradio-container { max-width: 100% !important; width: 100% !important; margin: 0 !important;
  padding: 0 clamp(16px, 3.5vw, 64px) !important; }
h1, h2, h3, .display { font-family: 'Josefin Sans', 'Work Sans', sans-serif; }

#hero { display:grid; grid-template-columns: minmax(0, 1.5fr) minmax(300px, 1fr); border-radius: 28px; overflow:hidden; margin: 8px 0 18px; }
#hero .copy { padding: clamp(28px, 3vw, 48px) clamp(20px, 2.5vw, 40px); background: var(--bg); }
#hero h1 { font-size: clamp(2.3rem, 4.4vw, 4.6rem); line-height: .96; margin: 0 0 18px; text-transform: uppercase; letter-spacing: .01em; }
#hero h1 .l1 { color: var(--ink); display:block; } #hero h1 .l2 { color: var(--ember); display:block; }
#hero p { font-size: 1.08rem; max-width: 54ch; line-height: 1.55; margin: 0; color: var(--ink) !important; }
#hero .steps { background: var(--steps-bg); padding: 30px clamp(24px, 3vw, 48px) 30px clamp(40px, 5vw, 72px);
  display:flex; flex-direction:column; justify-content:center; gap: 18px;
  clip-path: polygon(4% 0, 100% 0, 100% 100%, 0 100%, 3% 82%, 0 64%, 4% 45%, 1% 24%); }
#hero .step { display:grid; grid-template-columns: 44px 1fr; align-items:center; gap: 14px; }
#hero .step b { font-family:'Josefin Sans'; width:40px; height:40px; border-radius:50%; display:grid; place-items:center; font-size:1.1rem;
  background: var(--mustard); color: #0B2B3C !important; }
#hero .step span, #hero .step small { color: var(--steps-ink) !important; }
#hero .step span { line-height:1.35; font-size: 1.05rem; font-weight: 600; }
#hero .step small { display:block; opacity:.75; font-weight: 400; font-size: .9rem; }

.tabs button[role=tab] { font-family:'Josefin Sans'; font-size: 1.05rem; color: var(--ink) !important; }
.tabs button[role=tab][aria-selected=true] { color: var(--ember) !important; border-color: var(--ember) !important; }
button.primary, .primary { background: var(--ember) !important; color: #fff !important; border: none !important; border-radius: 9999px !important;
  font-family:'Josefin Sans'; font-size: 1.05rem !important; }
button.secondary { border-radius: 9999px !important; border: 2px solid var(--ink) !important; background: transparent !important; color: var(--ink) !important; }
button:focus-visible { outline: 3px solid var(--lagoon) !important; outline-offset: 2px; }

#meme { border: 6px solid var(--ember) !important; border-radius: 18px !important; background: var(--meme-bg) !important;
  box-shadow: 10px 10px 0 var(--mustard); }
.pill { display:inline-block; padding: 8px 16px; border-radius: 9999px; font-weight:600; }
.pill-ok { background: var(--lagoon); color: #fff !important; } .pill-no { background: var(--pill-no-bg); color: var(--pill-no-ink) !important; }
.fine { font-size: .92rem; opacity: .85; margin: 8px 0 0; color: var(--ink); }
.bars { display:flex; flex-direction:column; gap: 8px; margin-top: 4px; color: var(--ink); }
.bar { display:grid; grid-template-columns: 70px 1fr 44px; align-items:center; gap: 10px; }
.bar .track { height: 14px; background: var(--track); border-radius: 9999px; overflow:hidden; }
.bar .fill { display:block; height:100%; border-radius: 9999px; transition: width .5s ease; }
.dark .bar .fill.sad { background: #F2E8D5 !important; }   /* the navy "sad" bar would vanish on navy */
.bar.top .lab, .bar.top .val { font-weight: 700; }
@media (prefers-reduced-motion: reduce) { .bar .fill { transition: none; } }

#presets { background: var(--ember) !important; border-radius: 24px; padding: 18px 20px 10px; margin-top: 14px; border: none !important; }
#presets *:not(img) { background: transparent !important; border-color: transparent !important; color: #fff !important; }
#presets img { border-radius: 50% !important; border: 4px solid #fff; aspect-ratio: 1/1; object-fit: cover; }
#footer { display:grid; grid-template-columns: 1fr 1fr; border-radius: 24px; overflow:hidden; margin-top: 26px; }
#footer > div { padding: 22px 26px; line-height: 1.5; }
#footer .a, #footer .a * { background: var(--mustard); color: #0B2B3C !important; }
#footer .b, #footer .b * { background: var(--ember); color: #fff !important; }
#footer h3 { margin: 0 0 6px; font-size: 1.2rem; }
@media (max-width: 900px) { #hero, #footer { grid-template-columns: 1fr; } #hero .steps { clip-path: none; padding-left: 28px; } }
"""


def make_theme():
    P = PALETTE
    return gr.themes.Base(font=[gr.themes.GoogleFont("Work Sans"), "system-ui", "sans-serif"]).set(
        body_background_fill=P["sand"], body_background_fill_dark=P["midnight"],
        background_fill_primary=P["paper"], background_fill_primary_dark=P["panel"],
        background_fill_secondary=P["sand"], background_fill_secondary_dark=P["deep"],
        body_text_color=P["midnight"], body_text_color_dark=P["sand"],
        body_text_color_subdued="#4A6372", body_text_color_subdued_dark="#C9C0AE",
        block_background_fill=P["paper"], block_background_fill_dark=P["panel"],
        block_border_color="#E2D6BF", block_border_color_dark="#1E4A61",
        block_label_background_fill=P["paper"], block_label_background_fill_dark=P["panel"],
        block_label_text_color=P["midnight"], block_label_text_color_dark=P["sand"],
        block_title_text_color=P["midnight"], block_title_text_color_dark=P["sand"],
        input_background_fill="#F7F0E2", input_background_fill_dark=P["deep"],
        input_border_color="#E2D6BF", input_border_color_dark="#1E4A61",
        border_color_primary="#E2D6BF", border_color_primary_dark="#1E4A61",
        checkbox_label_background_fill="#F7F0E2", checkbox_label_background_fill_dark=P["deep"],
        checkbox_label_text_color=P["midnight"], checkbox_label_text_color_dark=P["sand"],
        color_accent_soft="#FBD9C4", color_accent_soft_dark="#3A2A1F",
    )


HERO = """
<section id="hero">
  <div class="copy">
    <h1><span class="l1">Read your dog's mood.</span><span class="l2">Make the meme.</span></h1>
    <p>Drop a dog photo, describe one, or do both. We check it's really a dog, read its emotion,
       and write the caption in Tounsi, English or French.</p>
  </div>
  <div class="steps">
    <div class="step"><b>1</b><span>Is it a dog?<small>YOLOv8 finds and crops the dog</small></span></div>
    <div class="step"><b>2</b><span>How does it feel?<small>Emotion model: angry, happy, relaxed or sad</small></span></div>
    <div class="step"><b>3</b><span>Make it funny<small>A language model writes the caption</small></span></div>
  </div>
</section>
"""


def footer():
    m = emotion.get_model().kind
    return f"""
<section id="footer">
  <div class="a"><h3>How sure is it?</h3>The emotion guess is right about 6 times out of 10 on test photos it never saw
  (64%), and it's least reliable for angry dogs. Dogs' faces are hard to read, even for people, so treat it as a fun
  read, not a diagnosis. The example photos above are ones it gets right.</div>
  <div class="b"><h3>Under the hood</h3>Dog check: YOLOv8s (Part 1) with a CLIP second opinion. Emotion: {html.escape(m)},
  trained on Part 1's data and compared with Part 2's models. Search and captions: CLIP and free language models,
  with offline templates as backup (Part 3).</div>
</section>"""


def build():
    with gr.Blocks(title="DogTranslator — dog mood memes", fill_width=True) as demo:
        gr.HTML(HERO)
        state = gr.State(None)
        hits_state = gr.State([])
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                lang = gr.Radio(LANG_CHOICES, value="darija", label="Caption language")
                use_llm = gr.Checkbox(value=True, label="Use the AI caption writer (falls back to templates offline)")
                with gr.Tabs(elem_classes="tabs"):
                    with gr.Tab("From a photo"):
                        img = gr.Image(type="pil", label="Your dog photo", sources=["upload", "clipboard", "webcam"], height=320)
                        idea = gr.Textbox(label="Joke idea (optional)", placeholder="e.g. Monday morning, exams, couscous on Sunday")
                        go_img = gr.Button("Make my meme", variant="primary")
                        with gr.Group(elem_id="presets"):
                            gr.Examples([[str(EX / f"{n}.jpg")] for n in ["angry", "happy", "relaxed", "sad", "cat"]],
                                        inputs=[img], label="Try one: four moods and a cat")
                    with gr.Tab("From a description"):
                        query = gr.Textbox(label="Describe the dog", placeholder="a grumpy pug judging you")
                        go_txt = gr.Button("Find a dog and make the meme", variant="primary")
                        gr.Examples([[q] for q in TEXT_EXAMPLES], inputs=[query], label="Ideas")
                        gallery = gr.Gallery(label="Closest dogs in our 13,238 photos (click one to use it)", columns=4, height=200,
                                             allow_preview=False)
            with gr.Column(scale=6):
                meme = gr.Image(type="pil", label="Your meme", elem_id="meme", height=460, interactive=False, format="jpeg")
                again = gr.Button("Another caption", variant="secondary")
                gate_box = gr.HTML()
                bars = gr.HTML()
                details = gr.HTML()
        gr.HTML(footer())

        outs = [meme, gate_box, bars, details, state]
        go_img.click(make_from_photo, [img, idea, lang, use_llm], outs)
        go_txt.click(search_and_make, [query, lang, use_llm], [gallery, hits_state, *outs])
        query.submit(search_and_make, [query, lang, use_llm], [gallery, hits_state, *outs])
        gallery.select(pick_photo, [query, lang, use_llm, hits_state], outs)
        again.click(new_caption, [state, lang, use_llm], [meme])
    return demo


if __name__ == "__main__":
    pipeline.warmup()
    build().launch(server_name="0.0.0.0", server_port=7860, theme=make_theme(), css=CSS, head=HEAD,
                  allowed_paths=[str(ROOT / "data" / "clean"), str(EX)])
