import torch
from app.model_loader import model, tokenizer, base_model
import re

VALID_ICD_CODES = {f"A{str(i).zfill(2)}" for i in range(100)}

def normalize_prediction(text: str):
    """
    Extract ICD-10 A00–A99 codes from model output.
    Preserves order and removes duplicates.
    """
    found = re.findall(r"A\d{2}", text.upper())
    unique = []
    for code in found:
        if code in VALID_ICD_CODES and code not in unique:
            unique.append(code)
    return unique

def predict_icd(clinical_text: str) -> str:
    prompt = (
        "Analyze the clinical description and predict the most relevant ICD-10 codes.\n"
        f"Clinical note:\n{clinical_text}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
    **inputs,
    max_length=32,
    do_sample=True,          # 🔥 KEY
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=3
)


    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Aggregate unique ICD codes across samples
    final_codes = []
    for d in decoded:
        for c in normalize_prediction(d):
            if c not in final_codes:
                final_codes.append(c)

    return final_codes[:3]
def general_medical_advice(clinical_text: str) -> str:
    """
    Uses the BASE model (no ICD constraints) to provide:
    - general medical context
    - possible concerns
    - first-aid / next-step guidance

    Output is plain text for frontend display.
    """

    prompt = (
    "You are a general medical assistant.\n\n"
    "Read the clinical description and respond in clear, natural language.\n"
    "Your response should:\n"
    "- Explain the possible health concern in simple terms\n"
    "- Suggest general first-aid or immediate care steps if appropriate\n"
    "- Advise when professional medical care should be sought\n\n"
    "Do not provide a medical diagnosis or prescribe medications.\n"
    "Keep the response concise, calm, and safety-focused.\n\n"
    f"Clinical description:\n{clinical_text}\n\n"
    "Response:"
)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = base_model.generate(
            **inputs,
            max_length=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()
