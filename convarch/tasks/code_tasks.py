"""Python function tasks graded by hidden unit tests.

Each test is a Python expression evaluated with the candidate function bound to `f`
and a helper `raises(exc_type, thunk)`. Every test is tagged with the shard whose
requirement it checks, so each non-intent shard is load-bearing.
"""

from .schema import Shard, Task

CONTEXT = "You are helping a software developer write Python code."

DELIVERABLE = (
    "Deliverable format: whenever you provide the code, give one complete, self-contained "
    "Python function (standard library only, imports inside the block) in a single "
    "```python fenced block. The function name must be exactly the one the user asked for."
)


def _t(shard, expr):
    return {"shard": shard, "expr": expr}


PARSE_DURATION = '''
import re

def parse_duration(s):
    if not isinstance(s, str):
        raise ValueError("not a string")
    t = s.strip().lower()
    sign = 1
    if t.startswith("-"):
        sign, t = -1, t[1:].strip()
    if not t:
        raise ValueError("empty")
    if re.fullmatch(r"\\d+(?:\\.\\d+)?", t):
        return sign * int(round(float(t)))
    mult = {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1}
    pos, total, found = 0, 0.0, False
    for m in re.finditer(r"\\s*(\\d+(?:\\.\\d+)?)\\s*([wdhms])\\s*", t):
        if m.start() != pos:
            raise ValueError("bad duration")
        total += float(m.group(1)) * mult[m.group(2)]
        pos, found = m.end(), True
    if not found or pos != len(t):
        raise ValueError("bad duration")
    return sign * int(round(total))
'''

NORMALIZE_PHONE = '''
import re

def normalize_phone(raw):
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    ext = None
    m = re.search(r"(?i)\\s*(?:x|ext\\.?|extension)\\s*(\\d+)\\s*$", s)
    if m:
        ext, s = m.group(1), s[:m.start()]
    if re.search(r"[A-Za-z]", s):
        return None
    digits = re.sub(r"\\D", "", s)
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    if len(digits) != 10 or digits[3:6] == "555":
        return None
    return "+1" + digits + (f";ext={ext}" if ext else "")
'''

MERGE_INTERVALS = '''
def merge_intervals(intervals):
    items = []
    for iv in intervals:
        if (not isinstance(iv, (list, tuple)) or len(iv) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in iv)):
            raise TypeError("each range must be a pair of numbers")
        a, b = iv
        if a > b:
            a, b = b, a
        if a == b:
            continue
        items.append((a, b))
    items.sort()
    out = []
    for a, b in items:
        if out and a < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
'''

TOP_WORDS = '''
import re
from collections import Counter

def top_words(text, k):
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower())
    words = [w for w in words if not w.isdigit() and len(w.replace("'", "")) >= 3]
    ranked = sorted(Counter(words).items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]
'''

ROLLING_MEAN = '''
def rolling_mean(values, window):
    if window < 1:
        raise ValueError("window must be >= 1")
    out = []
    for i in range(len(values)):
        if i < window - 1:
            out.append(None)
            continue
        chunk = [v for v in values[i - window + 1: i + 1] if v is not None and v <= 1000]
        out.append(round(sum(chunk) / len(chunk), 1) if chunk else None)
    return out
'''

SLUGIFY = '''
import re
import unicodedata

def slugify(title):
    s = title.replace("&", " and ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    words = [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in {"a", "an", "of"}]
    out = ""
    for w in words:
        cand = w if not out else out + "-" + w
        if len(cand) > 30:
            break
        out = cand
    return out
'''

TASKS = [
    Task(
        id="code_parse_duration",
        domain="code",
        title="Parse human duration strings",
        shards=[
            Shard(1, "need a python fn `parse_duration(s)` that turns strings like '1h30m' into a total number of seconds"),
            Shard(2, "It should support days (d), hours (h), minutes (m) and seconds (s)."),
            Shard(3, "Units can be upper or lower case, and there may be spaces between the parts, like '2D 4h'."),
            Shard(4, "If the string is just a bare number with no unit, treat it as minutes."),
            Shard(5, "If the string is empty or isn't a valid duration, raise a ValueError."),
            Shard(6, "Fractional amounts like '1.5h' are allowed; round the final total to the nearest whole second and return it as an int."),
            Shard(7, "Weeks too, written with 'w'."),
            Shard(8, "It's for the workout timer in our gym app.", distractor=True),
            Shard(9, "Actually, a bare number with no unit should mean seconds, not minutes.", revises=4,
                  full_text="If the string is just a bare number with no unit, treat it as seconds."),
            Shard(10, "A leading minus sign makes the whole duration negative, like '-1h30m'."),
        ],
        gold=PARSE_DURATION,
        grading={"entry": "parse_duration", "tests": [
            _t(1, "f('1h30m') == 5400"),
            _t(2, "f('2d') == 172800"),
            _t(2, "f('45s') == 45"),
            _t(2, "f('1d2h3m4s') == 93784"),
            _t(3, "f('2D 4h') == 187200"),
            _t(3, "f('1H 30M') == 5400"),
            _t(3, "f(' 10m 5s ') == 605"),
            _t(9, "f('90') == 90"),
            _t(9, "f('5') == 5"),
            _t(5, "raises(ValueError, lambda: f(''))"),
            _t(5, "raises(ValueError, lambda: f('abc'))"),
            _t(5, "raises(ValueError, lambda: f('5x'))"),
            _t(5, "raises(ValueError, lambda: f('-'))"),
            _t(6, "f('1.5h') == 5400"),
            _t(6, "f('0.75m') == 45"),
            _t(6, "f('2.4s') == 2 and type(f('2.4s')) is int"),
            _t(6, "f('2.4') == 2"),
            _t(7, "f('1w') == 604800"),
            _t(7, "f('1w2d') == 777600"),
            _t(10, "f('-1h30m') == -5400"),
            _t(10, "f('-90') == -90"),
        ]},
    ),
    Task(
        id="code_normalize_phone",
        domain="code",
        title="Normalize US phone numbers",
        shards=[
            Shard(1, "I need `normalize_phone(raw)` to clean up the phone numbers people type into our signup form."),
            Shard(2, "They're US numbers. The output should be E.164: a plus sign, country code 1, then the 10 digits with nothing else, like +14157770123."),
            Shard(3, "People type them every which way: with dashes, dots, spaces, or parentheses around the area code."),
            Shard(4, "Some include the leading country code (1 or +1) and some don't; both should work."),
            Shard(5, "If the number isn't valid (wrong number of digits, or it has letters in it), return None instead of raising an error."),
            Shard(6, "Some people add an extension at the end like 'x204' or 'ext. 204'. Just drop the extension and normalize the main number."),
            Shard(7, "Numbers with 555 as the exchange (the three digits right after the area code) are fake test numbers; return None for those."),
            Shard(8, "We're mostly a California business, if that matters.", distractor=True),
            Shard(9, "Actually, don't throw the extension away. Keep it by adding ';ext=' and its digits to the end, like +14157770123;ext=204.",
                  revises=6,
                  full_text="Some people add an extension at the end like 'x204' or 'ext. 204'. Keep it by appending ';ext=' and its digits to the normalized number, like +14157770123;ext=204."),
        ],
        gold=NORMALIZE_PHONE,
        grading={"entry": "normalize_phone", "tests": [
            _t(2, "f('4157770123') == '+14157770123'"),
            _t(3, "f('(415) 777-0123') == '+14157770123'"),
            _t(3, "f('415.777.0123') == '+14157770123'"),
            _t(3, "f('415 777 0123') == '+14157770123'"),
            _t(4, "f('1-415-777-0123') == '+14157770123'"),
            _t(4, "f('+1 (415) 777-0123') == '+14157770123'"),
            _t(5, "f('777-0123') is None"),
            _t(5, "f('415-777-01234') is None"),
            _t(5, "f('415-CALL-NOW') is None"),
            _t(9, "f('415-777-0123 x204') == '+14157770123;ext=204'"),
            _t(9, "f('(415) 777-0123 ext. 7') == '+14157770123;ext=7'"),
            _t(7, "f('415-555-0123') is None"),
            _t(7, "f('(212) 555 1234 x9') is None"),
        ]},
    ),
    Task(
        id="code_merge_intervals",
        domain="code",
        title="Merge overlapping time ranges",
        shards=[
            Shard(1, "write `merge_intervals(intervals)`: it takes a list of [start, end] ranges and merges the overlapping ones"),
            Shard(2, "The input isn't sorted; the output should be sorted by start."),
            Shard(3, "Ranges that just touch, like [1, 3] and [3, 5], count as overlapping and should merge into one range [1, 5]."),
            Shard(4, "Sometimes the data has start and end flipped, like [7, 4]; treat that as [4, 7]."),
            Shard(5, "Return a list of tuples (start, end), not lists."),
            Shard(6, "Drop zero-length ranges like [5, 5] before merging."),
            Shard(7, "If any item isn't a pair of numbers, raise a TypeError."),
            Shard(8, "They're meeting-room bookings from our calendar export.", distractor=True),
            Shard(9, "Actually, ranges that only touch, like [1, 3] and [3, 5], should stay separate. Only real overlaps get merged.",
                  revises=3,
                  full_text="Ranges that only touch, like [1, 3] and [3, 5], stay separate; only ranges that really overlap are merged."),
        ],
        gold=MERGE_INTERVALS,
        grading={"entry": "merge_intervals", "tests": [
            _t(1, "[tuple(x) for x in f([[1, 4], [2, 6]])] == [(1, 6)]"),
            _t(1, "[tuple(x) for x in f([[1.5, 2.5], [2, 3]])] == [(1.5, 3)]"),
            _t(1, "[tuple(x) for x in f([])] == []"),
            _t(2, "[tuple(x) for x in f([[8, 10], [1, 3]])] == [(1, 3), (8, 10)]"),
            _t(9, "[tuple(x) for x in f([[1, 3], [3, 5]])] == [(1, 3), (3, 5)]"),
            _t(9, "[tuple(x) for x in f([[1, 3], [2, 5], [5, 7]])] == [(1, 5), (5, 7)]"),
            _t(4, "[tuple(x) for x in f([[7, 4], [1, 2]])] == [(1, 2), (4, 7)]"),
            _t(5, "all(type(x) is tuple for x in f([[1, 2], [4, 6]])) and len(f([[1, 2], [4, 6]])) == 2"),
            _t(6, "[tuple(x) for x in f([[5, 5], [1, 2]])] == [(1, 2)]"),
            _t(6, "[tuple(x) for x in f([[3, 3]])] == []"),
            _t(7, "raises(TypeError, lambda: f([[1, 2, 3]]))"),
            _t(7, "raises(TypeError, lambda: f([['a', 'b']]))"),
        ]},
    ),
    Task(
        id="code_top_words",
        domain="code",
        title="Most frequent words",
        shards=[
            Shard(1, "write `top_words(text, k)` that gives me the k most common words in some text, with their counts"),
            Shard(2, "Counting should be case-insensitive ('The' and 'the' are the same word), and the words should be reported in lowercase."),
            Shard(3, "Ignore punctuation: strip it so 'word,' and 'word' count together. Apostrophes inside words, like don't, stay part of the word."),
            Shard(4, "Leave out these filler words entirely: the, a, an, and, of, to, in."),
            Shard(5, "If two words have the same count, order them alphabetically."),
            Shard(6, "Return a list of (word, count) tuples, most common first."),
            Shard(7, "Ignore words shorter than 3 letters."),
            Shard(8, "Numbers aren't words: skip tokens made only of digits."),
            Shard(9, "We'll run it over customer reviews.", distractor=True),
            Shard(10, "Actually, scrap the filler-word list. Don't exclude any words just for being filler words.",
                  revises=4, full_text="Don't exclude any words just for being filler words."),
        ],
        gold=TOP_WORDS,
        grading={"entry": "top_words", "tests": [
            _t(6, "f('cat dog cat', 1) == [('cat', 2)]"),
            _t(2, "[tuple(x) for x in f('Dog dog DOG cat', 1)] == [('dog', 3)]"),
            _t(3, "[tuple(x) for x in f('hello, world! hello.', 2)] == [('hello', 2), ('world', 1)]"),
            _t(3, "[tuple(x) for x in f(\"don't stop, don't\", 1)] == [(\"don't\", 2)]"),
            _t(5, "[tuple(x) for x in f('pear apple pear apple fig', 3)] == [('apple', 2), ('pear', 2), ('fig', 1)]"),
            _t(7, "[tuple(x) for x in f('ox ox ox cat', 1)] == [('cat', 1)]"),
            _t(8, "[tuple(x) for x in f('42 42 42 cats', 1)] == [('cats', 1)]"),
            _t(10, "[tuple(x) for x in f('the cat and the hat', 2)] == [('the', 2), ('and', 1)]"),
        ]},
    ),
    Task(
        id="code_rolling_mean",
        domain="code",
        title="Moving average with gaps",
        shards=[
            Shard(1, "need `rolling_mean(values, window)`, a moving average over a list of sensor readings"),
            Shard(2, "The output should be the same length as the input. Each position averages the window ending at that position; at the start, where a full window isn't available yet, average whatever readings exist so far."),
            Shard(3, "Some readings are missing and show up as None. Skip those and average only the readings that are present in the window."),
            Shard(4, "If a window has no readings at all, the output at that position should be None."),
            Shard(5, "Round each average to 2 decimal places."),
            Shard(6, "If window is less than 1, raise a ValueError."),
            Shard(7, "Readings above 1000 are sensor glitches; treat them like missing readings."),
            Shard(8, "The readings come from greenhouse temperature probes.", distractor=True),
            Shard(9, "Actually, at the start where there aren't `window` positions yet, output None instead of a partial average. The output still has the same length as the input.",
                  revises=2,
                  full_text="The output should be the same length as the input. Each position averages the window ending at that position; at the start, where fewer than `window` positions exist yet, output None."),
            Shard(10, "Round to 1 decimal place instead of 2.", revises=5,
                  full_text="Round each average to 1 decimal place."),
        ],
        gold=ROLLING_MEAN,
        grading={"entry": "rolling_mean", "tests": [
            _t(9, "f([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]"),
            _t(9, "f([3, 6, 9], 3) == [None, None, 6.0]"),
            _t(3, "f([1, None, 3, 5], 2) == [None, 1.0, 3.0, 4.0]"),
            _t(4, "f([None, None, 4], 2) == [None, None, 4.0]"),
            _t(10, "f([1, 2, 2, 2], 3) == [None, None, 1.7, 2.0]"),
            _t(6, "raises(ValueError, lambda: f([1, 2], 0))"),
            _t(7, "f([1, 5000, 3], 2) == [None, 1.0, 3.0]"),
            _t(7, "f([2000, 2000], 1) == [None, None]"),
        ]},
    ),
    Task(
        id="code_slugify",
        domain="code",
        title="URL slugs",
        shards=[
            Shard(1, "write `slugify(title)` to turn our article titles into URL slugs"),
            Shard(2, "Slugs are all lowercase, with words separated by single hyphens."),
            Shard(3, "Accented letters should become plain ASCII, like 'é' to 'e' and 'ñ' to 'n'."),
            Shard(4, "Anything that isn't a letter or a digit counts as a separator; runs of separators collapse into a single hyphen, and there are no hyphens at the start or end."),
            Shard(5, "Cap slugs at 40 characters, but never cut a word in half: drop whole words from the end until it fits."),
            Shard(6, "Actually, make the length cap 30 characters instead of 40.", revises=5,
                  full_text="Cap slugs at 30 characters, but never cut a word in half: drop whole words from the end until it fits."),
            Shard(7, "Drop these words from slugs: a, an, the, of."),
            Shard(8, "An '&' should turn into the word 'and'."),
            Shard(9, "Our blog runs on a static site generator.", distractor=True),
            Shard(10, "Actually, keep 'the' in slugs; only drop a, an and of.", revises=7,
                  full_text="Drop these words from slugs: a, an, of."),
        ],
        gold=SLUGIFY,
        grading={"entry": "slugify", "tests": [
            _t(2, "f('Hello World') == 'hello-world'"),
            _t(3, "f('Café Niño') == 'cafe-nino'"),
            _t(4, "f('  Rock -- Roll 2024!! ') == 'rock-roll-2024'"),
            _t(4, "f('C++ Tips') == 'c-tips'"),
            _t(8, "f('Rock & Roll') == 'rock-and-roll'"),
            _t(10, "f('A Tale of Two Cities') == 'tale-two-cities'"),
            _t(10, "f('The End of an Era') == 'the-end-era'"),
            _t(6, "f('The quick brown fox jumps over the lazy dog again') == 'the-quick-brown-fox-jumps-over'"),
            _t(6, "f('An Introduction of Python decorators for beginners') == 'introduction-python-decorators'"),
        ]},
    ),
]
