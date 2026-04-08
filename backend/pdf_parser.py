import sys, os
try:
    _BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BACKEND_DIR = os.getcwd()
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from pdf2image import convert_from_bytes
from model_loader import run_inference
from PIL import Image
import io

# Prompt engineering for datasheet parsing
TABLE_PROMPT = (
    "You are a precision semiconductor datasheet parser. "
    "Extract ALL technical data from this page including:\n"
    "- Electrical characteristics tables (with Parameter, Symbol, Min, Typ, Max, Unit columns)\n"
    "- Schematic block diagrams and their descriptions\n"
    "- Pin descriptions and configurations\n"
    "Output in clean Markdown. Use $LaTeX$ notation for all parameter symbols "
    "(e.g. $V_{GS(th)}$, $I_D$, $R_{DS(on)}$). "
    "If no technical data is present on this page, output: [NO_DATA]"
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


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 6) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL images (capped at max_pages)."""
    return convert_from_bytes(pdf_bytes, first_page=1, last_page=max_pages, dpi=150)


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
    images = pdf_to_images(pdf_bytes)
    pages_text = []
    for i, img in enumerate(images):
        print(f"[pdf_parser] Processing page {i + 1}/{len(images)}...")
        text = extract_text_from_page(img)
        if text:
            pages_text.append(f"### Page {i + 1}\n\n{text}")
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
