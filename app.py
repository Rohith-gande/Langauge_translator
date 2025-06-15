import gradio as gr
import torch
import tempfile
import os
import speech_recognition as sr
import gtts
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Language map
languages = {
    "ace_Arab": "Acehnese (Arabic script)", "ace_Latn": "Acehnese", "acm_Arab": "Mesopotamian Arabic",
    "acq_Arab": "Ta’izzi-Adeni Arabic", "aeb_Arab": "Tunisian Arabic", "afr_Latn": "Afrikaans",
    "ajp_Arab": "South Levantine Arabic", "aka_Latn": "Akan", "amh_Ethi": "Amharic",
    "apc_Arab": "North Levantine Arabic", "arb_Arab": "Modern Standard Arabic", "ars_Arab": "Najdi Arabic",
    "ary_Arab": "Moroccan Arabic", "arz_Arab": "Egyptian Arabic", "asm_Beng": "Assamese",
    "ast_Latn": "Asturian", "awa_Deva": "Awadhi", "ayr_Latn": "Aymara", "azj_Latn": "Azerbaijani",
    "bak_Cyrl": "Bashkir", "bam_Latn": "Bambara", "ban_Latn": "Balinese", "bel_Cyrl": "Belarusian",
    "bem_Latn": "Bemba", "ben_Beng": "Bengali", "bho_Deva": "Bhojpuri", "bjn_Arab": "Banjar (Arabic script)",
    "bjn_Latn": "Banjar", "bod_Tibt": "Tibetan", "bos_Latn": "Bosnian", "bug_Latn": "Buginese",
    "bul_Cyrl": "Bulgarian", "cat_Latn": "Catalan", "ceb_Latn": "Cebuano", "ces_Latn": "Czech",
    "cjk_Latn": "Chokwe", "ckb_Arab": "Central Kurdish", "crh_Latn": "Crimean Tatar", "cym_Latn": "Welsh",
    "dan_Latn": "Danish", "deu_Latn": "German", "dik_Latn": "Southwestern Dinka", "dyu_Latn": "Dyula",
    "dzo_Tibt": "Dzongkha", "ell_Grek": "Greek", "eng_Latn": "English", "epo_Latn": "Esperanto",
    "est_Latn": "Estonian", "eus_Latn": "Basque", "ewe_Latn": "Ewe", "fao_Latn": "Faroese",
    "fij_Latn": "Fijian", "fin_Latn": "Finnish", "fon_Latn": "Fon", "fra_Latn": "French",
    "fur_Latn": "Friulian", "fuv_Latn": "Nigerian Fulfulde", "gla_Latn": "Scottish Gaelic",
    "gle_Latn": "Irish", "glg_Latn": "Galician", "grn_Latn": "Guarani", "guj_Gujr": "Gujarati",
    "hat_Latn": "Haitian Creole", "hau_Latn": "Hausa", "heb_Hebr": "Hebrew", "hin_Deva": "Hindi",
    "hne_Deva": "Chhattisgarhi", "hrv_Latn": "Croatian", "hun_Latn": "Hungarian", "hye_Armn": "Armenian",
    "ibo_Latn": "Igbo", "ilo_Latn": "Ilocano", "ind_Latn": "Indonesian", "isl_Latn": "Icelandic",
    "ita_Latn": "Italian", "jav_Latn": "Javanese", "jpn_Jpan": "Japanese", "kab_Latn": "Kabyle",
    "kan_Knda": "Kannada", "kas_Deva": "Kashmiri (Devanagari script)", "kat_Geor": "Georgian",
    "kaz_Cyrl": "Kazakh", "khm_Khmr": "Khmer", "kin_Latn": "Kinyarwanda", "kor_Hang": "Korean",
    "lao_Laoo": "Lao", "lit_Latn": "Lithuanian", "lug_Latn": "Ganda", "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi", "mkd_Cyrl": "Macedonian", "mlg_Latn": "Malagasy", "mlt_Latn": "Maltese",
    "mni_Mtei": "Meitei", "msa_Latn": "Malay", "mya_Mymr": "Burmese", "nld_Latn": "Dutch",
    "npi_Deva": "Nepali", "ory_Orya": "Odia", "pan_Guru": "Punjabi", "pol_Latn": "Polish",
    "por_Latn": "Portuguese", "ron_Latn": "Romanian", "rus_Cyrl": "Russian", "san_Deva": "Sanskrit",
    "sin_Sinh": "Sinhala", "slk_Latn": "Slovak", "slv_Latn": "Slovenian", "spa_Latn": "Spanish",
    "sqi_Latn": "Albanian", "srp_Cyrl": "Serbian", "swa_Latn": "Swahili", "swe_Latn": "Swedish",
    "tam_Taml": "Tamil", "tel_Telu": "Telugu", "tha_Thai": "Thai", "tur_Latn": "Turkish",
    "ukr_Cyrl": "Ukrainian", "urd_Arab": "Urdu", "uzn_Latn": "Uzbek", "vie_Latn": "Vietnamese",
    "yor_Latn": "Yoruba", "zho_Hans": "Chinese (Simplified)", "zho_Hant": "Chinese (Traditional)"
}

lang_names = sorted(languages.values())
lang_code_map = {v: k for k, v in languages.items()}

def get_lang_code(name):
    return lang_code_map[name]

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tgt_token_id)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_text_interface(text, src_lang_name, tgt_lang_name):
    src_code = get_lang_code(src_lang_name)
    tgt_code = get_lang_code(tgt_lang_name)
    return translate(text, src_code, tgt_code)

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio."

def text_to_speech(text, lang_code="hi"):
    tts = gtts.gTTS(text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_fp:
        tts.save(tts_fp.name)
        sound = AudioSegment.from_mp3(tts_fp.name)
        wav_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sound.export(wav_fp.name, format="wav")
        return wav_fp.name

def translate_audio(audio_path, src_lang, tgt_lang):
    input_text = speech_to_text(audio_path)
    translated_text = translate(
        input_text,
        get_lang_code(src_lang),
        get_lang_code(tgt_lang)
    )
    tgt_gtts_lang = get_lang_code(tgt_lang).split("_")[0]
    try:
        tts_path = text_to_speech(translated_text, lang_code=tgt_gtts_lang)
    except Exception:
        tts_path = None
    return input_text, translated_text, tts_path

# Gradio interfaces
audio_interface = gr.Interface(
    fn=translate_audio,
    inputs=[
        gr.Audio(type="filepath", label="Speak Now"),
        gr.Dropdown(choices=lang_names, value="English", label="From Language"),
        gr.Dropdown(choices=lang_names, value="Hindi", label="To Language")
    ],
    outputs=[
        gr.Textbox(label="Recognized Input"),
        gr.Textbox(label="Translated Text"),
        gr.Audio(label="Spoken Translation", type="filepath")
    ],
    title="🎙️ Voice Language Translator",
    description="Speak and translate between 30+ languages using Meta's NLLB model."
)

text_interface = gr.Interface(
    fn=translate_text_interface,
    inputs=[
        gr.Textbox(label="Input Text"),
        gr.Dropdown(choices=lang_names, value="English", label="From Language"),
        gr.Dropdown(choices=lang_names, value="Hindi", label="To Language")
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="🌍 Universal Language Translator",
    description="Translate between 30+ languages using Meta's NLLB-200 model."
)

tabbed = gr.TabbedInterface(
    [text_interface, audio_interface],
    ["Text Translation", "Voice Translation"]
)

tabbed.launch()
