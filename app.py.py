"""
English → Urdu Legal Translation — Multi-Model Comparison Demo
Compares: mBART-50, NLLB-200, and a custom Transformer (Marian-based) model.

HOW TO USE:
1. Update MODEL_PATHS below with your Google Drive / local checkpoint paths
   (if using Colab: mount drive first with `from google.colab import drive; drive.mount('/content/drive')`)
2. pip install gradio transformers torch sentencepiece sacremoses
3. python app.py
"""

import spaces  # Must be imported before torch/transformers for ZeroGPU
import time
import torch
import gradio as gr
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    MarianMTModel, MarianTokenizer,
)

# ----------------------------------------------------------------------------
# 1. CONFIG — update these paths to your saved checkpoints
# ----------------------------------------------------------------------------
MODEL_PATHS = {
    "mBART-50": "areebanaz/mbart-legal-urdu",
    "NLLB-200": "areebanaz/nllb-legal-urdu",
    "Transformer (Marian)": "areebanaz/marian-legal-urdu",
}

DEVICE = "cuda"

MODEL_META = {
    "mBART-50": {"icon": "🟢", "tag": "Fine-tuned"},
    "NLLB-200": {"icon": "🔵", "tag": "Fine-tuned"},
    "Transformer (Marian)": {"icon": "🟣", "tag": "Custom"},
}

_loaded = {}  # cache so we don't reload on every request


def load_mbart(repo_id):
    tok = MBart50TokenizerFast.from_pretrained(repo_id)
    tok.src_lang = "en_XX"
    model = MBartForConditionalGeneration.from_pretrained(repo_id)
    model.to(DEVICE)
    model.eval()
    return tok, model


def load_nllb(repo_id):
    tok = AutoTokenizer.from_pretrained(repo_id, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    model.to(DEVICE)
    model.eval()
    return tok, model


def load_marian(repo_id):
    tok = MarianTokenizer.from_pretrained(repo_id)
    model = MarianMTModel.from_pretrained(repo_id)
    model.to(DEVICE)
    model.eval()
    return tok, model


LOADERS = {
    "mBART-50": load_mbart,
    "NLLB-200": load_nllb,
    "Transformer (Marian)": load_marian,
}


def get_model(name):
    if name not in _loaded:
        _loaded[name] = LOADERS[name](MODEL_PATHS[name])
    return _loaded[name]


# ZeroGPU expects GPU-dependent models to be placed on CUDA at module level.
# The decorator below makes the actual inference use a temporary GPU allocation.


# ----------------------------------------------------------------------------
# 2. TRANSLATION LOGIC
# ----------------------------------------------------------------------------
def translate_with(name, text):
    if not text.strip():
        return "", 0.0

    start = time.time()
    try:
        tok, model = get_model(name)

        with torch.inference_mode():
            if name == "mBART-50":
                enc = tok(text, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                gen = model.generate(
                    **enc,
                    forced_bos_token_id=tok.lang_code_to_id["ur_PK"],
                    max_length=200,
                )
                out = tok.batch_decode(gen, skip_special_tokens=True)[0]

            elif name == "NLLB-200":
                enc = tok(text, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                gen = model.generate(
                    **enc,
                    forced_bos_token_id=tok.convert_tokens_to_ids("urd_Arab"),
                    max_length=200,
                )
                out = tok.batch_decode(gen, skip_special_tokens=True)[0]

            else:  # Marian / custom transformer
                enc = tok(text, return_tensors="pt", padding=True)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                gen = model.generate(**enc, max_length=200)
                out = tok.batch_decode(gen, skip_special_tokens=True)[0]

    except Exception as e:
        out = f"⚠️ Model error ({e.__class__.__name__}): {e}"

    elapsed = round(time.time() - start, 2)
    return out, elapsed


@spaces.GPU(duration=120)
def translate_all(text):
    results = []
    for name in MODEL_PATHS:
        out, t = translate_with(name, text)
        results.append(out)
        results.append(f"⏱ {t}s")
    return results


# ----------------------------------------------------------------------------
# 3. UI  —  professional legal-tech theme (deep navy + gold accent)
# ----------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#080d18",
    body_background_fill_dark="#080d18",
    block_background_fill="#101827",
    block_background_fill_dark="#101827",
    block_border_color="#26344d",
    block_border_color_dark="#26344d",
    block_radius="16px",
    block_label_text_color="#b7c3d9",
    block_title_text_color="#f3f6fb",
    body_text_color="#edf2f8",
    body_text_color_subdued="#9aa9c1",
    input_background_fill="#0b1322",
    input_background_fill_dark="#0b1322",
    input_border_color="#2b3b57",
    input_border_color_focus="#c9a227",
    button_primary_background_fill="#159a83",
    button_primary_background_fill_hover="#117d6b",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#182238",
    button_secondary_text_color="#e2bd45",
    button_secondary_border_color="#c9a227",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Nastaliq+Urdu&display=swap');

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: #080d18 !important;
}

#hero {
    text-align: center;
    padding: 32px 22px 25px 22px;
    border-radius: 18px;
    margin-bottom: 20px;
    background: linear-gradient(145deg, #111c31 0%, #0e1728 100%);
    border: 1px solid #2b3b57;
    box-shadow: 0 10px 30px rgba(0,0,0,.22);
}

#hero h1 {
    font-size: 30px;
    font-weight: 700;
    margin: 0;
    color: #f1d477 !important;
    letter-spacing: .2px;
}

#hero p {
    color: #aab8ce;
    margin-top: 10px;
    font-size: 14.5px;
    max-width: 660px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.55;
}

#hero .pillrow {
    margin-top: 16px;
    display: flex;
    justify-content: center;
    gap: 9px;
    flex-wrap: wrap;
}

.pill {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: #121e32;
    border: 1px solid #354662;
    color: #e3c35b;
}

#input-card {
    border-radius: 16px !important;
    border: 1px solid #2b3b57 !important;
    background: #101827 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.24);
    padding: 5px;
}

#input-card textarea {
    background: #0b1322 !important;
    color: #f2f5f9 !important;
    border-color: #2b3b57 !important;
}

#translate-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: .2px;
    background: #159a83 !important;
}

#translate-btn:hover {
    background: #117d6b !important;
}

#clear-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
    background: #182238 !important;
    color: #e2bd45 !important;
    border: 1px solid #34445f !important;
}

.section-title {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #edf2f8 !important;
    margin: 22px 0 12px 2px !important;
}

.model-card {
    padding: 15px !important;
    border-radius: 16px !important;
    border: 1px solid #2b3b57 !important;
    background: #101827 !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.24) !important;
    transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}

.model-card:hover {
    transform: translateY(-2px);
    border-color: #c9a227 !important;
    box-shadow: 0 12px 28px rgba(0,0,0,.30) !important;
}

/* Important: force model names to stay clearly visible in every Gradio theme. */
.model-name {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    min-height: 30px !important;
    margin-bottom: 9px !important;
    padding: 0 2px !important;
    font-family: Poppins, ui-sans-serif, system-ui, sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    line-height: 1.3 !important;
    color: #f8fafc !important;
    opacity: 1 !important;
}

.model-name .model-title {
    color: #f8fafc !important;
    opacity: 1 !important;
    white-space: nowrap !important;
}

.model-tag {
    margin-left: auto !important;
    flex-shrink: 0 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    color: #e4c55f !important;
    background: rgba(201,162,39,.10) !important;
    border: 1px solid rgba(201,162,39,.42) !important;
    padding: 3px 8px !important;
    border-radius: 999px !important;
}

.model-card .urdu-output textarea {
    font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
    font-size: 19px !important;
    line-height: 2.1 !important;
    color: #f4f7fb !important;
    background: #0b1322 !important;
    border: 1px solid #273751 !important;
}

.model-card .urdu-output {
    background: transparent !important;
}

.badge {
    font-size: 11.5px !important;
    color: #8392aa !important;
    text-align: right !important;
    margin-top: 4px !important;
}

#footnote {
    text-align: center;
    color: #71819b;
    font-size: 12px;
    margin-top: 26px;
    padding-top: 18px;
    border-top: 1px solid #26344d;
}

footer { visibility: hidden; }

@media (max-width: 800px) {
    #hero h1 { font-size: 24px; }
    .model-name { font-size: 14px !important; }
}
"""

EXAMPLES = [
    "The court hereby directs the respondent to comply with the order within thirty days.",
    "This appeal is dismissed for lack of merit and the impugned judgment is upheld.",
    "The petitioner is entitled to compensation as determined by the trial court.",
]

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="EN→UR Legal Translation") as demo:

    gr.HTML(
        """
        <div id="hero">
            <h1>⚖️ English → Urdu Legal Translation</h1>
            <p>A side-by-side comparison of three fine-tuned neural translation models — trained on
            Pakistani Supreme Court judgments — for accurate, domain-aware legal translation.</p>
            <div class="pillrow">
                <span class="pill">mBART-50</span>
                <span class="pill">NLLB-200</span>
                <span class="pill">Transformer (Marian)</span>
                <span class="pill">20k+ Legal Sentence Pairs</span>
            </div>
        </div>
        """
    )

    with gr.Group(elem_id="input-card"):
        with gr.Row():
            input_box = gr.Textbox(
                label="✍️ English legal text",
                placeholder="Type or paste an English legal sentence here...",
                lines=4,
                scale=4,
            )
            with gr.Column(scale=1, min_width=150):
                translate_btn = gr.Button("🔁  Translate", variant="primary", size="lg", elem_id="translate-btn")
                clear_btn = gr.ClearButton(value="✖  Clear", elem_id="clear-btn")

    gr.Examples(examples=EXAMPLES, inputs=input_box, label="💡 Try an example")

    gr.Markdown("### 📑 Model Outputs", elem_classes="section-title")

    with gr.Row():
        outputs = {}
        timers = {}
        for name in MODEL_PATHS:
            meta = MODEL_META[name]
            with gr.Column(elem_classes="model-card", variant="panel"):
                gr.HTML(
                    f"""<div class="model-name">
                            <span class="model-title">{meta['icon']} {name}</span>
                            <span class="model-tag">{meta['tag']}</span>
                        </div>"""
                )
                outputs[name] = gr.Textbox(
                    label="Urdu translation", rtl=True, lines=4, interactive=False,
                    elem_classes="urdu-output",
                )
                timers[name] = gr.Markdown("", elem_classes="badge")

    gr.HTML(
        """
        <div id="footnote">
            Trained on ~20,000 English–Urdu parallel legal sentence pairs &nbsp;·&nbsp;
            Evaluated with BLEU, METEOR, TER, chrF++, BERTScore
        </div>
        """
    )

    def run_all(text):
        raw = translate_all(text)
        texts = raw[0::2]
        times = [f"⏱ {t}" if isinstance(t, str) else t for t in raw[1::2]]
        return texts + times

    all_outputs = list(outputs.values()) + list(timers.values())
    translate_btn.click(fn=run_all, inputs=input_box, outputs=all_outputs)
    clear_btn.add([input_box] + list(outputs.values()) + list(timers.values()))

if __name__ == "__main__":
    demo.launch()
