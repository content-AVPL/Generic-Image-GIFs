import io
import json
import re
import time
from PIL import Image, ImageEnhance, ImageChops
import streamlit as st
from google import genai
from google.genai import types
from streamlit_cropper import st_cropper
from config import GEMINI_API_KEY


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Generic Images",
    layout="wide",
    page_icon="🎨",
)

st.markdown(
    "<h1 style='text-align:center; margin-bottom:1.0rem;'>Generic Images & GIFs</h1>",
    unsafe_allow_html=True,
)


# ---------- GENAI CLIENT ----------
client = genai.Client(api_key=GEMINI_API_KEY)


# ---------- SESSION STATE ----------
if "images" not in st.session_state:
    st.session_state.images = []
if "crops" not in st.session_state:
    st.session_state.crops = []
if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "character_desc" not in st.session_state:
    st.session_state.character_desc = ""
if "env_desc" not in st.session_state:
    st.session_state.env_desc = ""
if "char_animals_json" not in st.session_state:
    st.session_state.char_animals_json = {}
if "char_animals_text" not in st.session_state:
    st.session_state.char_animals_text = ""
if "gif_enabled" not in st.session_state:
    st.session_state.gif_enabled = False
if "gif_count" not in st.session_state:
    st.session_state.gif_count = 3
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "story"  # "story" or "image_desc"
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Dark"


# ============================================================
# INPUT MODE + TEXT AREAS
# ============================================================
c_story_chk, c_img_chk = st.columns(2)

with c_story_chk:
    story_enabled = st.checkbox("Use story description", value=True, key="chk_story")
with c_img_chk:
    img_enabled = st.checkbox("Use direct image descriptions", value=False, key="chk_img")

# enforce mutual exclusivity (prefer image descriptions if both toggled)
if story_enabled and img_enabled:
    st.session_state.chk_story = False
    story_enabled = False

st.session_state.input_mode = "image_desc" if img_enabled else "story"

c_story, c_img_desc = st.columns(2)

with c_story:
    story_text = st.text_area(
        "Story description",
        height=150,
        placeholder="Example: Two friends Meera and Ravi take their brown dog to the park...",
        key="story_text",
        disabled=not story_enabled,
    )

with c_img_desc:
    image_desc_text = st.text_area(
        "Direct image descriptions (one sentence per image)",
        height=150,
        placeholder=(
            "1. Meera and Ravi are packing their school bags in the classroom.\n"
            "2. They walk with their brown dog in the sunny park.\n"
            "3. They read books together under a big tree..."
        ),
        key="image_desc_text",
        disabled=not img_enabled,
    )


# ============================================================
# OPTIONS: GENRE, FONT, THEME, NUM IMAGES, GIF CONTROLS
# ============================================================
c_genre_font, c_theme, c_imgs, c_gif, c_submit = st.columns([1.3, 1.3, 1, 1.4, 1])

with c_genre_font:
    genre = st.selectbox(
        "Visual Genre",
        [
            "Cartoonistic",
            "Flat classroom illustration",
            "Storybook style",
            "Wimmelbuch Style",
            "Whimsical Illustrations Style",
            "Realistic Illustration Style",
            "Abstract Illustration Style",
            "Moody Illustrations",
            "Line Drawing and Sketch Style",
            "Vintage Style Illustrations",
            "Comic strip style",
        ],
    )
    font_size = st.slider("UI Font size", 10, 28, 16)

# Theme definitions
THEMES = {
    "Dark": {
        "bg": "#0e1117",
        "text": "#f8f9fa",
        "secondary_bg": "#1f2933",
        "primary": "#ff4b4b",
    },
    "Light": {
        "bg": "#ffffff",
        "text": "#111111",
        "secondary_bg": "#ffffff",
        "primary": "#ff4b4b",
    },
    "Forest green": {
        "bg": "#081810",
        "text": "#e9fff2",
        "secondary_bg": "#0f2b1a",
        "primary": "#3dd68c",
    },
    "Classic gray": {
        "bg": "#4e4444",
        "text": "#111111",
        "secondary_bg": "#e0e0e0",
        "primary": "#4b8cf5",
    },
    "Sunset": {
        "bg": "#B16E30",
        "text": "#ffe7f2",
        "secondary_bg": "#d87f69",
        "primary": "#ff7a59",
    },
    "Ocean blue": {
        "bg": "#3996f3",
        "text": "#e6f4ff",
        "secondary_bg": "#052033",
        "primary": "#38bdf8",
    },
    "Sand": {
        "bg": "#fdf5e6",
        "text": "#3a2f1b",
        "secondary_bg": "#f3e2bf",
        "primary": "#c27a3f",
    },
}

with c_theme:
    st.session_state.ui_theme = st.selectbox(
        "UI theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.ui_theme),
        key="ui_theme_sel",
    )

with c_imgs:
    num_images = st.selectbox("Number of images", [1, 2, 3, 4, 5, 6, 7, 8], index=0)

with c_gif:
    st.session_state.gif_enabled = st.checkbox("GIFs?", value=False, key="gif_enable_chk")
    if st.session_state.gif_enabled:
        st.session_state.gif_count = st.selectbox(
            "GIF count",
            options=[1, 2, 3],
            index=2,
            help="How many GIF columns to show (max 3).",
            key="gif_count_sel",
        )
    else:
        st.session_state.gif_count = 0

with c_submit:
    submit = st.button("Generate Images", type="primary", use_container_width=True)


# ============================================================
# THEME CSS (strong overrides)
# ============================================================
if st.session_state.ui_theme not in THEMES:
    st.session_state.ui_theme = "Light"

theme_cfg = THEMES[st.session_state.ui_theme]
bg = theme_cfg["bg"]
text = theme_cfg["text"]
secondary_bg = theme_cfg["secondary_bg"]
primary = theme_cfg["primary"]

st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {primary};
        --text-color: {text};
        --bg-color: {bg};
        --secondary-bg-color: {secondary_bg};
    }}

    html, body {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}

    .stApp, .stAppViewContainer, .main, .block-container {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}

    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, label, span, p, li {{
        color: var(--text-color) !important;
    }}

    textarea, input[type="text"], input[type="number"], .stTextInput input {{
        background-color: var(--secondary-bg-color) !important;
        color: var(--text-color) !important;
        border-radius: 4px;
    }}

    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"],
    .stSlider, .stCheckbox, .stRadio {{
        color: var(--text-color) !important;
    }}

    .stButton > button {{
        background-color: var(--primary-color) !important;
        color: #ffffff !important;
        border-radius: 4px;
        border: none;
    }}
    .stButton > button:hover {{
        opacity: 0.9;
    }}

    .stDownloadButton > button {{
        background-color: var(--primary-color) !important;
        color: #ffffff !important;
        border-radius: 4px;
        border: none;
    }}

    [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
        background-color: var(--secondary-bg-color) !important;
        color: var(--text-color) !important;
    }}

    html, body, [class*="css"] {{
        font-size: {font_size}px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- HELPERS ----------
def image_bytes_for_format(img, fmt):
    buf = io.BytesIO()
    if fmt == "JPEG":
        if img.mode == "RGBA":
            bg_ = Image.new("RGB", img.size, (255, 255, 255))
            bg_.paste(img, mask=img.split()[3])
            bg_.save(buf, format="JPEG")
        else:
            img.convert("RGB").save(buf, format="JPEG")
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def autocrop_border(im, tolerance=10, min_crop_ratio=0.95):
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    w, h = im.size
    bg_color = im.getpixel((0, 0))
    bg_ = Image.new(im.mode, im.size, bg_color)
    diff = ImageChops.difference(im, bg_)
    diff = ImageChops.add(diff, diff, 2.0, -tolerance)
    bbox = diff.getbbox()
    if not bbox:
        return im
    cropped = im.crop(bbox)
    cw, ch = cropped.size
    if (cw / float(w) < min_crop_ratio) or (ch / float(h) < min_crop_ratio):
        return cropped
    return im


def call_image_model(prompt, debug_label="", ref_image=None):
    try:
        contents = []
        if ref_image is not None:
            buf = io.BytesIO()
            ref_image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                )
            )
        contents.append(prompt)
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )
        if resp is None or getattr(resp, "candidates", None) is None or len(resp.candidates) == 0:
            st.error(f"[{debug_label}] Image model returned no candidates.")
            return None
        data = None
        for cand in resp.candidates:
            if cand is None or getattr(cand, "content", None) is None:
                continue
            for part in cand.content.parts:
                inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    break
            if data:
                break
        if not data:
            st.error(f"[{debug_label}] No image data found in response parts.")
            return None
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        img = autocrop_border(img)
        return img
    except Exception as e:
        st.error(f"[{debug_label}] Error calling image model: {e}")
        return None


def split_story_into_scenes(story, n_scenes):
    prompt = f"""
You are a storyboard artist.

Split this story into {n_scenes} short scene descriptions in order.
Each scene description should be 1–2 sentences, visual, and mention the same main characters.

Return output exactly in this JSON format:
{{"scenes": ["scene1 text", "scene2 text", "..."]}}

Story:
\"\"\"{story}\"\"\"
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )
    text = resp.text
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        parts = [p.strip() for p in story.split(".") if p.strip()]
        if not parts:
            return [story] * n_scenes
        chunk_size = max(1, len(parts) // n_scenes)
        scenes = []
        for i in range(0, len(parts), chunk_size):
            scenes.append(". ".join(parts[i:i + chunk_size]))
        while len(scenes) < n_scenes:
            scenes.append(scenes[-1])
        return scenes[:n_scenes]
    data = json.loads(m.group(0))
    scenes = data.get("scenes", [])
    if not scenes:
        return [story] * n_scenes
    while len(scenes) < n_scenes:
        scenes.append(scenes[-1])
    return scenes[:n_scenes]


def extract_environment_description(story):
    prompt = f"""
Read this story and infer the main environment where most of the events happen
(for example: classroom, park, kitchen, bedroom).

Describe:
- The setting (indoor/outdoor, type of place, time of day if clear)
- Colors and style of background elements (walls, floor, sky, trees, etc.)
- Any important animals (just mention them briefly; details will be in a separate list)
- Any important objects (chairs, tables, bottles, balls, books, etc.) with colors.

Return 3–5 sentences, very concrete, with exact colors and details.
Do not mention multiple different environments.
Return only the description text.

Story:
\"\"\"{story}\"\"\"
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )
    return resp.text.strip()


def extract_characters_and_animals(story):
    prompt = f"""
You are preparing a storyboard for an illustrated children's book.

Read this story and identify all recurring important human characters (children and adults)
and all important animals.

Return JSON ONLY in this exact format (no extra text):

{{
  "humans": [
    {{
      "id": "child1",
      "role": "main child",
      "short_label": "Meera",
      "description": "A shy 8-year-old girl with long straight black hair in two ponytails, medium brown skin, big dark eyes, and a yellow T-shirt with blue jeans and red sneakers."
    }}
  ],
  "animals": [
    {{
      "id": "animal1",
      "species": "dog",
      "short_label": "the dog",
      "description": "A small brown dog with floppy ears, a white patch on its chest, and a bright blue collar with a round silver tag."
    }}
  ]
}}

Rules:
- Include every human or animal that appears more than once or is important to the story.
- Each description must be 1–3 sentences and very specific about age, body type, hair/fur color and style,
  skin/fur pattern, eye color, outfit or collar, and any ornaments (belts, hats, glasses, bows, etc.).
- Use stable IDs like "child1", "child2", "adult1", "animal1", "animal2".
- If there is only one human or one animal, still wrap it in a list.
- If there are no animals, use "animals": [].
- Output must be valid JSON compatible with json.loads in Python.

Story:
\"\"\"{story}\"\"\"
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )
    text = resp.text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"humans": [], "animals": []}, "No structured character data."
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {"humans": [], "animals": []}, "Could not parse character JSON."
    humans = data.get("humans", [])
    animals = data.get("animals", [])
    lines = []
    if humans:
        lines.append("Human characters (draw ALL of them in every image unless the scene clearly excludes someone):")
        for h in humans:
            label = h.get("short_label", h.get("id", "human"))
            desc = h.get("description", "")
            lines.append(f"- {label}: {desc}")
    if animals:
        lines.append("Animals (draw ALL of them whenever they logically belong in the scene):")
        for a in animals:
            label = a.get("short_label", a.get("id", "animal"))
            desc = a.get("description", "")
            lines.append(f"- {label}: {desc}")
    text_block = "\n".join(lines) if lines else "No recurring humans or animals were detected."
    return {"humans": humans, "animals": animals}, text_block


def build_entity_count_sentence(char_animals):
    humans = char_animals.get("humans", [])
    animals = char_animals.get("animals", [])
    child_count = sum(1 for h in humans if "child" in (h.get("role", "").lower()))
    adult_count = len(humans) - child_count
    parts = []
    if child_count > 0:
        parts.append(f"{child_count} children")
    if adult_count > 0:
        parts.append(f"{adult_count} adults")
    species_counts = {}
    for a in animals:
        sp = (a.get("species") or "animal").lower()
        species_counts[sp] = species_counts.get(sp, 0) + 1
    for sp, cnt in species_counts.items():
        if cnt == 1:
            parts.append(f"1 {sp}")
        else:
            parts.append(f"{cnt} {sp}s")
    if not parts:
        return "Draw the main character(s) from the story."
    return (
        "Draw exactly " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] +
        " in each image, unless the scene description clearly says otherwise. "
        "Do not add extra people or animals, and do not remove any that should be visible."
    )


# ---------- MAIN GENERATION PIPELINE ----------
if submit:
    mode = st.session_state.input_mode  # "story" or "image_desc"

    if mode == "story" and not story_text.strip():
        st.warning("Story mode is selected. Please enter a story.")
    elif mode == "image_desc" and not image_desc_text.strip():
        st.warning("Image description mode is selected. Please enter image descriptions.")
    else:
        if mode == "image_desc":
            raw_lines = [ln.strip() for ln in image_desc_text.splitlines() if ln.strip()]
            if len(raw_lines) < num_images:
                st.warning("Number of image description lines is less than selected number of images.")
            scenes = (raw_lines + [raw_lines[-1]] * num_images)[:num_images]
            base_text_for_extraction = image_desc_text
        else:
            with st.spinner(f"Splitting story into {num_images} scenes..."):
                scenes = split_story_into_scenes(story_text, num_images)
            base_text_for_extraction = story_text

        st.session_state.scenes = scenes

        with st.spinner("Extracting environment from text..."):
            env_desc = extract_environment_description(base_text_for_extraction)
            st.session_state.env_desc = env_desc

        with st.spinner("Extracting characters and animals..."):
            char_animals_json, char_animals_text = extract_characters_and_animals(base_text_for_extraction)
            st.session_state.char_animals_json = char_animals_json
            st.session_state.char_animals_text = char_animals_text

        humans = st.session_state.char_animals_json.get("humans", [])
        if humans:
            st.session_state.character_desc = humans[0].get("description", "")
        else:
            st.session_state.character_desc = ""

        count_sentence = build_entity_count_sentence(st.session_state.char_animals_json)

        style_block = f"""
Art style: {genre} for K-3 students, flat cartoon, simple shapes, bright colors,
soft outlines, classroom-safe. All images must share the same art style,
color palette, camera angle (mid-shot, eye level), and character design.

Use one single consistent environment for all images based on this description:
{st.session_state.env_desc}

If the environment looks like a classroom, park, kitchen, or similar in the first image,
keep that exact environment in all later images: same wall colors, floor type,
windows, furniture layout, and lighting.

Important objects such as chairs, bottles, balls, tables, books, school bags, etc.
must keep the exact same color, pattern, and approximate position relative to the characters
across all images.

The illustration must be a full-bleed image: the scene fills the entire frame,
with no white margins, frames, or decorative borders on any side.
"""

        entity_block = f"""
Characters and animals that must stay visually consistent across all images:

{st.session_state.char_animals_text}

{count_sentence}

For every image:
- Keep each person's face, hairstyle, hair color, skin tone, body type, and outfit exactly the same.
- Keep each animal's species, fur color and pattern, size, collar/belt color, and any ornaments exactly the same.
- Only change poses, gestures, and facial expressions to match the scene.
"""

        safety_lead = """
Generate a safe, classroom-friendly cartoon illustration for K-3 children.
No realistic photos, no violence, no sensitive or adult topics.
"""

        with st.spinner(f"Generating {num_images} images from the text..."):
            imgs = []

            # Image 1
            first_prompt = f"""
Story Image 1 (Scene 1).

{safety_lead}

Scene: {scenes[0]}

{entity_block}

Make this the reference image for the entire sequence. Fix the appearance of all humans and animals
and the environment layout and key object colors.

Draw a full-bleed scene that clearly shows all required characters and animals in the main environment.

{style_block}
"""
            img1 = call_image_model(first_prompt, debug_label="Image 1")
            imgs.append(img1)
            time.sleep(5)

            # Images 2..N
            for idx in range(1, num_images):
                i = idx + 1
                scene = scenes[idx]
                p = f"""
Story Image {i} (Scene {i}).

{safety_lead}

You are continuing a storyboard. Use Image 1 as the visual reference.

{entity_block}

Preserve identities of all humans and animals and keep the environment and object colors identical.
Only poses and expressions may change.

Scene description for this panel:
{scene}

Draw a full-bleed image that matches the style and environment of Image 1.

{style_block}
"""
                img = call_image_model(p, debug_label=f"Image {i}", ref_image=img1)
                if img is None:
                    st.warning(f"Image {i} could not be generated.")
                imgs.append(img)
                time.sleep(5)

            st.session_state.images = imgs
            st.session_state.crops = [None] * len(imgs)


# --------- DISPLAY SECTIONS ----------
if st.session_state.scenes:
    st.markdown("#### Main Environment")
    st.markdown(st.session_state.env_desc)
    st.markdown("#### Characters & Animals")
    st.markdown(f"``````")
    st.markdown("#### Scenes")
    for i, sc in enumerate(st.session_state.scenes, start=1):
        st.markdown(f"**Scene {i}:** {sc}")


# Crop modes
ar_map = {
    "Free hand": None,
    "16:9": (16, 9),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "9:16": (9, 16),
    "3:2": (3, 2),
    "2:3": (2, 3),
}


st.markdown("---")
st.markdown("### Images")

images = st.session_state.images
crops = st.session_state.crops

if not images:
    st.info("Generate images to see panels.")
else:
    n = len(images)
    for row_start in range(0, n, 3):
        row_imgs = images[row_start:row_start + 3]
        row_cols = st.columns(len(row_imgs), gap="medium")
        for j, img in enumerate(row_imgs):
            idx = row_start + j
            with row_cols[j]:
                st.markdown(f"**Image {idx+1} – Scene {idx+1}**")
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    crop_on = st.checkbox("Crop", key=f"cropchk_{idx}")
                with c2:
                    crop_mode = st.selectbox(
                        "Crop mode",
                        list(ar_map.keys()),
                        index=0,
                        key=f"cropmode_{idx}",
                    )
                with c3:
                    fmt = st.selectbox(
                        "Format",
                        ["PNG", "JPEG", "GIF", "WEBP"],
                        index=0,
                        key=f"fmt_{idx}",
                    )
                if img is not None:
                    base_img = images[idx]
                    if crop_on:
                        st.info(f"Drag box to crop Image {idx+1}; click outside when done.")
                        aspect_tuple = ar_map.get(crop_mode)
                        cropped = st_cropper(
                            base_img,
                            aspect_ratio=aspect_tuple,
                            box_color="red",
                            return_type="image",
                            key=f"cropper_{idx}",
                        )
                        if cropped is not None:
                            crops[idx] = autocrop_border(cropped)
                    img_to_show = crops[idx] or base_img
                    st.image(img_to_show, caption=f"Image {idx+1} Preview", use_column_width=True)
                    data = image_bytes_for_format(img_to_show, fmt)
                    st.download_button(
                        "Download image",
                        data,
                        file_name=f"image{idx+1}.{fmt.lower()}",
                        mime=f"image/{fmt.lower()}",
                        use_container_width=True,
                        key=f"dl_{idx}",
                    )
                else:
                    st.info(f"Image {idx+1} was not generated.")


# ---------- GIF helpers ----------
def align_translation(frames, mode="RGBA"):
    max_w = max(im.width for im in frames)
    max_h = max(im.height for im in frames)
    aligned = []
    for im in frames:
        base = Image.new(mode, (max_w, max_h), (0, 0, 0, 0))
        base.paste(im, (0, 0))
        aligned.append(base)
    return aligned


def align_euclidean(frames, mode="RGBA"):
    max_w = max(im.width for im in frames)
    max_h = max(im.height for im in frames)
    aligned = []
    for im in frames:
        base = Image.new(mode, (max_w, max_h), (0, 0, 0, 0))
        x = (max_w - im.width) // 2
        y = (max_h - im.height) // 2
        base.paste(im, (x, y))
        aligned.append(base)
    return aligned


def align_similarity(frames, mode="RGBA"):
    max_w = max(im.width for im in frames)
    max_h = max(im.height for im in frames)
    aligned = []
    for im in frames:
        w, h = im.size
        scale = min(max_w / float(w), max_h / float(h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = im.resize((new_w, new_h), Image.LANCZOS)
        base = Image.new(mode, (max_w, max_h), (0, 0, 0, 0))
        x = (max_w - new_w) // 2
        y = (max_h - new_h) // 2
        base.paste(resized, (x, y))
        aligned.append(base)
    return aligned


def align_affine(frames, mode="RGBA"):
    return align_similarity(frames, mode)


# ---------- GIF section (conditional) ----------
if st.session_state.gif_enabled and st.session_state.gif_count > 0:
    st.markdown("---")
    st.markdown("### GIFs")

    available_imgs = []
    for c, i in zip(st.session_state.crops, st.session_state.images):
        if i is None:
            continue
        base = c or i
        base = autocrop_border(base)
        available_imgs.append(base)

    if not available_imgs:
        st.info("Generate images first to create GIFs.")
    else:
        count = min(st.session_state.gif_count, 3)
        gif_cols = st.columns(count, gap="large")

        # GIF A
        if count >= 1:
            with gif_cols[0]:
                st.markdown("#### GIF A – Euclidean")
                a1, a2, a3 = st.columns(3)
                with a1:
                    gA_speed = st.slider("Speed", 1, 10, 1, key="gA_speed")
                with a2:
                    gA_loop = st.checkbox("Loop", value=True, key="gA_loop")
                with a3:
                    gA_align = st.selectbox(
                        "Alignment",
                        ["Euclidean", "Translation", "Affine"],
                        index=0,
                        key="gA_align",
                    )
                b1, b2, b3 = st.columns(3)
                with b1:
                    gA_bright = st.slider("Bright", 0.5, 1.5, 1.0, 0.05, key="gA_bright")
                with b2:
                    gA_sat = st.slider("Sat", 0.5, 1.5, 1.0, 0.05, key="gA_sat")
                with b3:
                    gA_contrast = st.slider("Contr", 0.5, 1.5, 1.0, 0.05, key="gA_contrast")

                gA_frames = []
                for im in available_imgs:
                    f = im.convert("RGBA")
                    f = ImageEnhance.Brightness(f).enhance(gA_bright)
                    f = ImageEnhance.Color(f).enhance(gA_sat)
                    f = ImageEnhance.Contrast(f).enhance(gA_contrast)
                    gA_frames.append(f)

                if gA_align == "Euclidean":
                    aligned_A = align_euclidean(gA_frames)
                elif gA_align == "Translation":
                    aligned_A = align_translation(gA_frames)
                else:
                    aligned_A = align_affine(gA_frames)

                bufA = io.BytesIO()
                durationA = int(1000 / gA_speed)
                aligned_A[0].save(
                    bufA,
                    format="GIF",
                    save_all=True,
                    append_images=aligned_A[1:],
                    duration=durationA,
                    loop=0 if gA_loop else 1,
                    disposal=2,
                )
                gifA_bytes = bufA.getvalue()
                st.image(gifA_bytes, caption=f"GIF A ({gA_align})", use_column_width=True)
                st.download_button(
                    f"Download GIF A ({gA_align})",
                    gifA_bytes,
                    file_name=f"story_{gA_align.lower()}_A.gif",
                    mime="image/gif",
                    use_container_width=True,
                    key="dl_gifA",
                )

        # GIF B
        if count >= 2:
            with gif_cols[1]:
                st.markdown("#### GIF B – Translation")
                c1, c2, c3 = st.columns(3)
                with c1:
                    gB_speed = st.slider("Speed ", 1, 10, 1, key="gB_speed")
                with c2:
                    gB_loop = st.checkbox("Loop ", value=True, key="gB_loop")
                with c3:
                    gB_align = st.selectbox(
                        "Alignment ",
                        ["Translation", "Euclidean", "Affine"],
                        index=0,
                        key="gB_align",
                    )
                d1, d2, d3 = st.columns(3)
                with d1:
                    gB_bright = st.slider("Bright ", 0.5, 1.5, 1.0, 0.05, key="gB_bright")
                with d2:
                    gB_sat = st.slider("Sat ", 0.5, 1.5, 1.0, 0.05, key="gB_sat")
                with d3:
                    gB_contrast = st.slider("Contr ", 0.5, 1.5, 1.0, 0.05, key="gB_contrast")

                gB_frames = []
                for im in available_imgs:
                    f = im.convert("RGBA")
                    f = ImageEnhance.Brightness(f).enhance(gB_bright)
                    f = ImageEnhance.Color(f).enhance(gB_sat)
                    f = ImageEnhance.Contrast(f).enhance(gB_contrast)
                    gB_frames.append(f)

                if gB_align == "Translation":
                    aligned_B = align_translation(gB_frames)
                elif gB_align == "Euclidean":
                    aligned_B = align_euclidean(gB_frames)
                else:
                    aligned_B = align_affine(gB_frames)

                bufB = io.BytesIO()
                durationB = int(1000 / gB_speed)
                aligned_B[0].save(
                    bufB,
                    format="GIF",
                    save_all=True,
                    append_images=aligned_B[1:],
                    duration=durationB,
                    loop=0 if gB_loop else 1,
                    disposal=2,
                )
                gifB_bytes = bufB.getvalue()
                st.image(gifB_bytes, caption=f"GIF B ({gB_align})", use_column_width=True)
                st.download_button(
                    f"Download GIF B ({gB_align})",
                    gifB_bytes,
                    file_name=f"story_{gB_align.lower()}_B.gif",
                    mime="image/gif",
                    use_container_width=True,
                    key="dl_gifB",
                )

        # GIF C
        if count >= 3:
            with gif_cols[2]:
                st.markdown("#### GIF C – Affine")
                e1, e2, e3 = st.columns(3)
                with e1:
                    gC_speed = st.slider("Speed  ", 1, 10, 1, key="gC_speed")
                with e2:
                    gC_loop = st.checkbox("Loop  ", value=True, key="gC_loop")
                with e3:
                    gC_align = st.selectbox(
                        "Alignment  ",
                        ["Affine", "Euclidean", "Translation"],
                        index=0,
                        key="gC_align",
                    )
                f1, f2, f3 = st.columns(3)
                with f1:
                    gC_bright = st.slider("Bright  ", 0.5, 1.5, 1.0, 0.05, key="gC_bright")
                with f2:
                    gC_sat = st.slider("Sat  ", 0.5, 1.5, 1.0, 0.05, key="gC_sat")
                with f3:
                    gC_contrast = st.slider("Contr  ", 0.5, 1.5, 1.0, 0.05, key="gC_contrast")

                gC_frames = []
                for im in available_imgs:
                    f = im.convert("RGBA")
                    f = ImageEnhance.Brightness(f).enhance(gC_bright)
                    f = ImageEnhance.Color(f).enhance(gC_sat)
                    f = ImageEnhance.Contrast(f).enhance(gC_contrast)
                    gC_frames.append(f)

                if gC_align == "Affine":
                    aligned_C = align_affine(gC_frames)
                elif gC_align == "Euclidean":
                    aligned_C = align_euclidean(gC_frames)
                else:
                    aligned_C = align_translation(gC_frames)

                bufC = io.BytesIO()
                durationC = int(1000 / gC_speed)
                aligned_C[0].save(
                    bufC,
                    format="GIF",
                    save_all=True,
                    append_images=aligned_C[1:],
                    duration=durationC,
                    loop=0 if gC_loop else 1,
                    disposal=2,
                )
                gifC_bytes = bufC.getvalue()
                st.image(gifC_bytes, caption=f"GIF C ({gC_align})", use_column_width=True)
                st.download_button(
                    f"Download GIF C ({gC_align})",
                    gifC_bytes,
                    file_name=f"story_{gC_align.lower()}_C.gif",
                    mime="image/gif",
                    use_container_width=True,
                    key="dl_gifC",
                )
