#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."

def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        print("key: "+ key, "value: "+ value);
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_amounts(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    amounts = []
    for entry in value:
        try:
            amounts.append(abs(float(entry)))
        except (TypeError, ValueError):
            continue
    return amounts


def discount_query_total(data: dict[str, Any]) -> float:
    """query2 for one receipt: SUBTOTAL + every discount line (never ROUNDING)."""
    subtotal = as_float(data.get("subtotal"))
    return subtotal + sum(as_amounts(data.get("discounts")))


def item_query_total(data: dict[str, Any]) -> float:
    """Independent query2 estimate: the sum of all positive item prices."""
    total = 0.0
    for item in data.get("items", []) or []:
        if isinstance(item, dict):
            total += as_float(item.get("price"))
    return total


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """

    # 1. instantiate the DeepSeek vision model (load DEEPSEEK_API_KEY from the .env file)
    deepseek_llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0.0,
    )

    # 2. define Prompt template (include instructions for Hong Kong Receipt Calculation Rules)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise OCR extractor for Hong Kong supermarket receipts. Read the receipt image and extract these raw fields. Do NOT do any addition yourself; just report the numbers you see.

- "subtotal": the number on the SUBTOTAL line. This is the amount AFTER all discounts and BEFORE rounding.
- "discounts": an array containing the POSITIVE amount of EVERY discount / promotion / coupon / value-off / percentage-off line (use the absolute value of that line). Do NOT include the SUBTOTAL line, any item line, the payment line, or the ROUNDING line.
- "rounding": the number on the ROUNDING line (it is usually negative, e.g. -0.01). If there is no rounding line, use 0.
- "payment": the final amount actually paid, i.e. the number on the payment line (OCTOPUS / Visa / Cash / etc.).
- "items": an array of every positive-priced item line as {{"name": "...", "price": <number>}}. Skip discount lines.

Rules:
- The SUBTOTAL already has the discounts removed, so query2 = subtotal + sum(discounts). List discounts as POSITIVE numbers.
- NEVER put the ROUNDING line into "discounts".

Example (receipt5):
  Items: 028499 BAKERY 5  $10.00
         089288 咸蛋蒸肉餅  $36.90
         013723 急凍螺子肉   $60.80
  5% OFF: -$5.39
  SUBTOTAL: $102.31
  ROUNDING: -$0.01
  OCTOPUS: $102.30

Output JSON (this receipt's values):
{{"subtotal": 102.31, "discounts": [5.39], "rounding": -0.01, "payment": 102.30, "items": [{{"name": "028499 BAKERY 5", "price": 10.00}}, {{"name": "089288 咸蛋蒸肉餅", "price": 36.90}}, {{"name": "013723 急凍螺子肉", "price": 60.80}}]}}

Output ONLY a single valid JSON object. No explanation, no markdown, no extra characters."""),
        ("human", [
            {"type": "text", "text": "Please process this receipt image:"},
            {"type": "image_url", "image_url": {"url": "{image_url}"}}
        ])
    ])

    # 3. Parser: Convert JSON text generated by LLM into Python dict.
    parser = JsonOutputParser()

    # 4. Assemble the LCEL chain (Prompt -> LLM -> Parser)
    base_chain = prompt | deepseek_llm | parser

    # 5. Verification prompt: re-read the receipt when the two independent
    #    query2 estimates disagree (usually a missed discount line).
    verify_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are auditing a Hong Kong supermarket receipt extraction that FAILED a consistency check.

query2 can be computed two equivalent ways and they must match:
  (a) SUBTOTAL + the sum of every discount/promotion/coupon line (as positive numbers)
  (b) the sum of every positive item price
If (a) != (b), one line was missed or misread. The most common cause is a MISSED discount line.

Re-read the receipt image carefully and fix it:
- "discounts" MUST list EVERY discount / promotion / coupon / value-off / percentage-off line as a POSITIVE number.
- NEVER include the ROUNDING line, the SUBTOTAL line, the payment line, or any positive item in "discounts".
- Add any missing item line to "items" and any missing discount line to "discounts".

Return the corrected JSON in the SAME format as before: keys "subtotal", "discounts", "rounding", "payment", "items".
Output ONLY a single valid JSON object. No explanation, no markdown, no extra characters."""),
        ("human", [
            {"type": "text", "text": "Consistency check that failed:\n{report}\n\nPlease re-read the receipt image and return the corrected JSON."},
            {"type": "image_url", "image_url": {"url": "{image_url}"}}
        ])
    ])
    verify_chain = verify_prompt | deepseek_llm | parser

    def extract_and_verify(inputs: dict[str, Any]) -> Any:
        data = base_chain.invoke(inputs)
        if not isinstance(data, dict):
            return data

        subtotal_plus = discount_query_total(data)
        item_sum = item_query_total(data)
        mismatch = subtotal_plus > 0 and item_sum > 0 and abs(subtotal_plus - item_sum) > 0.005
        if mismatch:
            report = (
                f"previous_extraction = {json.dumps(data, ensure_ascii=False)}\n"
                f"SUBTOTAL + discounts = {subtotal_plus:.2f}\n"
                f"sum(item prices)     = {item_sum:.2f}"
            )
            fixed = verify_chain.invoke(
                {"image_url": inputs["image_url"], "report": report}
            )
            if isinstance(fixed, dict) and fixed.get("subtotal") is not None:
                return fixed
        return data

    return RunnableLambda(extract_and_verify)


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    batch_inputs = [
        {"image_url": image_data_url(img_path)} for img_path in images
    ]

    # 2. Use LangChain's batch method to extract data from each receipt in parallel.
    results = chain.batch(batch_inputs)

    # 3. Iterate over the extracted results and aggregate.
    total_query_1 = 0.0
    total_query_2 = 0.0

    for res in results:
        if not isinstance(res, dict):
            continue

        subtotal = as_float(res.get("subtotal"))
        discounts = as_amounts(res.get("discounts"))
        rounding = as_float(res.get("rounding"))
        payment = as_float(res.get("payment"))

        # query2 = SUBTOTAL + every discount line added back as a positive number
        # (the ROUNDING line is never added back).
        query_2 = subtotal + sum(discounts)
        # query1 = the final payment after the ROUNDING line.
        query_1 = payment if payment else subtotal + rounding
        total_query_1 += query_1
        total_query_2 += query_2
    # 4. Return the dictionary after aggregating the entire folder in the format required by the template.
    return {
        QUERY_1: f"{total_query_1:.2f}",
        QUERY_2: f"{total_query_2:.2f}"
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
