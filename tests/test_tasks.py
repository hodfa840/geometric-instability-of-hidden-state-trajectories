import pytest

from vane.tasks import bbh, cruxeval, gsm8k, humaneval, math500


def test_gsm8k_extraction():
    assert gsm8k.extract_answer("so 3 + 4 = 7\n#### 7") == "7"
    assert gsm8k.extract_answer("The answer: 1,234") == "1234"
    assert gsm8k.extract_answer("first 3 then 18") == "18"
    assert gsm8k.score("... #### 42", {"gt": "42"}) == ("42", 1)
    assert gsm8k.score("no number", {"gt": "42"}) == (None, 0)


def test_math500_boxed():
    text = r"so the result is \boxed{\frac{1}{2}} and done"
    assert math500.extract_boxed(text) == r"\frac{1}{2}"
    assert math500.score(text, {"gt": r"\frac{1}{2}"})[1] == 1
    assert math500.score(r"\boxed{x^{2}}", {"gt": "x^2"})[1] == 1   # {2} -> 2


def test_cruxeval_output():
    assert cruxeval.extract_answer("trace...\nOutput: [1,2 , 3]") == "[1, 2, 3]"
    assert cruxeval.is_correct(cruxeval.extract_answer('Output: "abc"'), "'abc'") == 1
    assert cruxeval.is_correct("abc", "'abc'") == 1                 # dropped quotes
    assert cruxeval.extract_answer("Output:\n```\n42\n```") == "42"


def test_bbh_matching():
    assert bbh.score("reasoning...\nAnswer: (B)", {"gt": "(B)"})[1] == 1
    assert bbh.score("Answer: **No**", {"gt": "No"})[1] == 1
    assert bbh.score("Answer: a) the quail", {"gt": "(A)"})[1] == 1
    assert bbh.score("Answer: (C)", {"gt": "(A)"})[1] == 0


def test_humaneval_execution(monkeypatch):
    item = {"prompt": "def add(a, b):\n", "entry_point": "add",
            "test": "def check(f):\n    assert f(2, 3) == 5\n"}
    good = "Sure.\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    with pytest.raises(RuntimeError):
        humaneval.score(good, item)                     # disabled by default
    monkeypatch.setattr(humaneval, "ALLOW_EXECUTION", True)
    assert humaneval.score(good, item)[1] == 1
    assert humaneval.score(bad, item)[1] == 0
