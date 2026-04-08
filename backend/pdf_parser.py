import sys, os
try:
    _BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BACKEND_DIR = os.getcwd()
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import io
import pypdfium2 as pdfium       # pure Python, no poppler system dep
from PIL import Image
from model_loader import run_inference

# Prompt engineering for datasheet parsing
TABLE_PROMPT = (
    "You are a precision engineering datasheet parser. "
    "Extract ALL technical data from this page including:\n"
    "- ALL tables (specifications, dimensions, characteristics)\n"
    "- Features, descriptions, and applications\n"
    "- Diagrams and package outlines (describe what they show)\n"
    "Output in clean Markdown. Represent all tables accurately using Markdown table syntax. "
    "Use $LaTeX$ for any mathematical symbols or units (e.g. $10 \\Omega$, $ppm/^\\circ C$). "
    "If the page is entirely blank or contains absolutely no technical information, output exactly: [NO_DATA]"
)

CHAT_PROMPT_TEMPLATE = (
    "You are a highly precise electronics engineering assistant. "
    "Answer questions using ONLY the provided datasheet context below.\n\n"
    "## Datasheet Context\n{context}\n\n"
    "## Question\n{query}\n\n"
    "## Instructions\n"
    "- Present values in a clean Markdown table if the question asks for parameters.\n"
    "- Use $LaTeX$ for all electrical symbols.\n"
    "- If the answer is not in the context, say: 'This information is not available in the uploaded datasheet.'\n"
    "- Do NOT hallucinate values."
)

SPEC_EXTRACTION_PROMPT = (
    "From the following datasheet content, extract these three values if present. "
    "Return ONLY a JSON object with keys: Vgs, Id, Rdson. "
    "Values should be strings including units (e.g. '2.5 V', '30 A', '5.0 mΩ'). "
    "Use 'N/A' if not found.\n\nContent:\n{content}"
)


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 6, dpi: int = 96) -> list:
    """
    Convert PDF bytes to a list of PIL Images using pypdfium2.
    No poppler / system dependency required.
    DPI=96 keeps images small enough for T4 VRAM budget.
    """
    doc = pdfium.PdfDocument(pdf_bytes)
    images = []
    n_pages = min(len(doc), max_pages)
    scale = dpi / 72  # pypdfium2 base is 72 DPI

    for i in range(n_pages):
        page = doc[i]
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()

        # Cap the largest dimension at 1024px to stay safely within VRAM
        max_dim = 1024
        if max(pil_image.size) > max_dim:
            pil_image.thumbnail((max_dim, max_dim), Image.LANCZOS)

        images.append(pil_image)

    doc.close()
    return images


def extract_text_from_page(image: Image.Image) -> str:
    """Run vision inference on a single page image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TABLE_PROMPT},
            ],
        }
    ]
    result = run_inference(messages, max_new_tokens=4096)
    return result if result.strip() != "[NO_DATA]" else ""


def extract_all_pages(pdf_bytes: bytes) -> str:
    """Process all pages and concatenate the extracted Markdown."""
    import torch
    images = pdf_to_images(pdf_bytes)
    pages_text = []
    for i, img in enumerate(images):
        print(f"[pdf_parser] Processing page {i + 1}/{len(images)}...")
        text = extract_text_from_page(img)
        if text:
            pages_text.append(f"### Page {i + 1}\n\n{text}")
        # Free VRAM between pages to avoid accumulation
        torch.cuda.empty_cache()
    return "\n\n---\n\n".join(pages_text)


def generate_chat_response(context: str, query: str) -> str:
    """Run LLM inference for a user query against the parsed context."""
    prompt = CHAT_PROMPT_TEMPLATE.format(context=context[:12000], query=query)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return run_inference(messages, max_new_tokens=2048)


def extract_specs(context: str) -> dict:
    """Extract the three key specs for the sidebar from datasheet context."""
    import json, re
    prompt = SPEC_EXTRACTION_PROMPT.format(content=context[:4000])
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    raw = run_inference(messages, max_new_tokens=256)
    # Find JSON block in the response
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"Vgs": "N/A", "Id": "N/A", "Rdson": "N/A"}
