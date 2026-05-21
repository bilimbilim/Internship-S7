import language_tool_python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics")

tool = language_tool_python.LanguageTool("fr")


# Score orthographe
def spelling_score(text):
    matches = tool.check(text)
    errors = len(matches)
    words = len(text.split())

    return round(max(0, 1 - errors / max(words, 1)), 2)


# Score structure email
def structure_score(text):
    score = 0

    if "Bonjour" in text:
        score += 1
    if "Cordialement" in text:
        score += 1
    if len(text.split()) > 30:
        score += 1

    return score / 3


def email_quality_score(text):

    s1 = spelling_score(text)
    s2 = structure_score(text)
    score = round((s1 + s2) / 2, 2)

    logger.info(f"Score IA -> Orthographe={s1}, Structure={s2}, Global={score}")

    return score
