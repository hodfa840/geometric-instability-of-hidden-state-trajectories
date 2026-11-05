"""GSM8K (Cobbe et al., 2021), test split, 1,319 problems."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

NAME = "gsm8k"

SYSTEM = "You are a math reasoning assistant. Think step by step."

FEW_SHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after the grove workers planted more. So there must have been 21 - 15 = 6 trees planted.\n#### 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. Then 2 more arrive. 3 + 2 = 5.\n#### 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.\n#### 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. He got 2 from his mom and 2 from his dad, so he got 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys.\n#### 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
     "From Monday to Thursday is 4 days. 5 computers were installed each day, so 5 * 4 = 20 computers were added. There were 9 originally, so now there are 9 + 20 = 29.\n#### 29"),
    ("Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?",
     "Michael started with 58 golf balls. He lost 23 on Tuesday, so he had 58 - 23 = 35. Then he lost 2 more on Wednesday, so he had 35 - 2 = 33.\n#### 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. She bought 5 bagels for 3 dollars each, spending 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.\n#### 8"),
]


def extract_answer(text: str) -> Optional[str]:
    """``#### n`` marker, then ``Answer: n``, then the last integer in the text."""
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"[Aa]nswer\s*[:\s]\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def load(max_samples: Optional[int] = None, **_) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    return [{"idx": i, "question": ds[i]["question"], "gt": extract_answer(ds[i]["answer"])} for i in range(n)]


def messages(item: dict):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["question"]}]
    shots = [f"Q: {q}\nA: {a}\n" for q, a in FEW_SHOT] + [f"Q: {item['question']}\nA:"]
    return msgs, "\n".join(shots)


def score(text: str, item: dict) -> Tuple[Optional[str], int]:
    pred = extract_answer(text)
    return pred, int(pred is not None and pred == item["gt"])
