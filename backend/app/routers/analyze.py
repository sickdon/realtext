import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AnalyzeRequest, AnalyzeResponse, MetricsResponse, SuggestionResponse,
    HumanizeRequest, HumanizeResponse,
)
from app.services.language_detector import detect_language
from app.services.text_analyzer import analyze_text
from app.services.llm_advisor import get_llm_suggestions
from app.services.llm_humanizer import humanize_text
from app.services.rate_limiter import check_rate_limit
from app.db.database import log_analysis

logger = logging.getLogger(__name__)

router = APIRouter()

FALLBACK_SUGGESTIONS: dict[str, dict[str, dict[str, str]]] = {
    "burstiness": {
        "it": {
            "issue": "Le tue frasi hanno una lunghezza troppo uniforme, tipica dei testi generati da AI.",
            "suggestion": "Prova ad alternare frasi molto corte (3-5 parole) con frasi piu lunghe (20+ parole). Inserisci domande retoriche o esclamazioni.",
        },
        "en": {
            "issue": "Your sentences have very uniform length, which is typical of AI-generated text.",
            "suggestion": "Try alternating very short sentences (3-5 words) with longer ones (20+ words). Add rhetorical questions or exclamations.",
        },
        "es": {
            "issue": "Tus oraciones tienen una longitud muy uniforme, tipica de textos generados por IA.",
            "suggestion": "Intenta alternar oraciones muy cortas (3-5 palabras) con otras mas largas (20+ palabras). Agrega preguntas retoricas o exclamaciones.",
        },
    },
    "lexical": {
        "it": {
            "issue": "Il vocabolario utilizzato e poco vario. L'AI tende a riutilizzare le stesse parole.",
            "suggestion": "Usa sinonimi, espressioni colloquiali o gergali. Evita di ripetere le stesse parole in frasi vicine.",
        },
        "en": {
            "issue": "The vocabulary used is not very diverse. AI tends to reuse the same words.",
            "suggestion": "Use synonyms, colloquial expressions, or slang. Avoid repeating the same words in nearby sentences.",
        },
        "es": {
            "issue": "El vocabulario utilizado es poco variado. La IA tiende a reutilizar las mismas palabras.",
            "suggestion": "Usa sinonimos, expresiones coloquiales o jerga. Evita repetir las mismas palabras en oraciones cercanas.",
        },
    },
    "repetition": {
        "it": {
            "issue": "Il testo contiene connettivi e strutture tipiche dell'AI (inoltre, in conclusione, e importante notare...).",
            "suggestion": "Elimina i connettivi formali. Inizia le frasi in modo diverso: con un'azione, un dettaglio concreto, o un pensiero diretto.",
        },
        "en": {
            "issue": "The text contains connectives and structures typical of AI (furthermore, in conclusion, it is important to note...).",
            "suggestion": "Remove formal connectives. Start sentences differently: with an action, a concrete detail, or a direct thought.",
        },
        "es": {
            "issue": "El texto contiene conectores y estructuras tipicas de la IA (ademas, en conclusion, es importante destacar...).",
            "suggestion": "Elimina los conectores formales. Comienza las oraciones de manera diferente: con una accion, un detalle concreto o un pensamiento directo.",
        },
    },
    "sentence_var": {
        "it": {
            "issue": "La struttura delle frasi e troppo regolare e prevedibile.",
            "suggestion": "Mescola frasi dichiarative, interrogative e esclamative. Usa incisi tra trattini o parentesi per rompere il ritmo.",
        },
        "en": {
            "issue": "The sentence structure is too regular and predictable.",
            "suggestion": "Mix declarative, interrogative, and exclamatory sentences. Use parenthetical asides or dashes to break the rhythm.",
        },
        "es": {
            "issue": "La estructura de las oraciones es demasiado regular y predecible.",
            "suggestion": "Mezcla oraciones declarativas, interrogativas y exclamativas. Usa incisos entre guiones o parentesis para romper el ritmo.",
        },
    },
}

SUMMARY_TEMPLATES: dict[str, dict[str, str]] = {
    "it": {
        "low": "Il testo mostra caratteristiche fortemente associate alla scrittura AI. Segui i consigli per renderlo piu naturale.",
        "mid": "Il testo ha alcune caratteristiche tipiche dell'AI. Con qualche modifica puo sembrare piu naturale.",
        "good": "Il testo sembra abbastanza naturale. Piccoli aggiustamenti possono migliorarlo ulteriormente.",
        "human": "Il testo appare scritto da un essere umano. Stile naturale, buona variazione e nessun pattern tipico dell'AI rilevato.",
    },
    "en": {
        "low": "The text shows characteristics strongly associated with AI writing. Follow the suggestions to make it more natural.",
        "mid": "The text has some AI-typical characteristics. A few changes can make it sound more natural.",
        "good": "The text seems fairly natural. Small adjustments can improve it further.",
        "human": "The text appears to be human-written. Natural style, good variation, and no typical AI patterns detected.",
    },
    "es": {
        "low": "El texto muestra caracteristicas fuertemente asociadas con la escritura de IA. Sigue los consejos para hacerlo mas natural.",
        "mid": "El texto tiene algunas caracteristicas tipicas de la IA. Con algunos cambios puede sonar mas natural.",
        "good": "El texto parece bastante natural. Pequenos ajustes pueden mejorarlo aun mas.",
        "human": "El texto parece escrito por un humano. Estilo natural, buena variacion y ningun patron tipico de IA detectado.",
    },
}


def _get_fallback_suggestions(metrics: dict, language: str, score: int) -> list[dict]:
    # If score is high (>= 70), text looks human — no negative suggestions needed
    if score >= 70:
        return []

    lang = language if language in ("it", "en", "es") else "en"
    suggestions: list[dict] = []
    idx = 0

    if metrics.get("burstiness_score", 50) < 55:
        fb = FALLBACK_SUGGESTIONS["burstiness"][lang]
        suggestions.append({
            "paragraph_index": idx,
            "original_excerpt": "",
            "issue": fb["issue"],
            "suggestion": fb["suggestion"],
            "impact": "high",
        })
        idx += 1

    if metrics.get("repetition_score", 50) < 55:
        fb = FALLBACK_SUGGESTIONS["repetition"][lang]
        suggestions.append({
            "paragraph_index": idx,
            "original_excerpt": "",
            "issue": fb["issue"],
            "suggestion": fb["suggestion"],
            "impact": "high",
        })
        idx += 1

    if metrics.get("lexical_diversity", 50) < 55:
        fb = FALLBACK_SUGGESTIONS["lexical"][lang]
        suggestions.append({
            "paragraph_index": idx,
            "original_excerpt": "",
            "issue": fb["issue"],
            "suggestion": fb["suggestion"],
            "impact": "medium",
        })
        idx += 1

    if metrics.get("sentence_length_variance", 50) < 55:
        fb = FALLBACK_SUGGESTIONS["sentence_var"][lang]
        suggestions.append({
            "paragraph_index": idx,
            "original_excerpt": "",
            "issue": fb["issue"],
            "suggestion": fb["suggestion"],
            "impact": "medium",
        })
        idx += 1

    return suggestions


def _get_summary(score: int, language: str) -> str:
    lang = language if language in ("it", "en", "es") else "en"
    templates = SUMMARY_TEMPLATES[lang]
    if score <= 30:
        return templates["low"]
    elif score <= 50:
        return templates["mid"]
    elif score <= 70:
        return templates["good"]
    else:
        return templates["human"]


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    if request.client:
        return request.client.host
    return "unknown"


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, request: Request) -> AnalyzeResponse | JSONResponse:
    client_ip = _get_client_ip(request)

    allowed, remaining = await check_rate_limit(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again tomorrow."},
            headers={"X-RateLimit-Remaining": "0"},
        )

    logger.info("Analyze request, text length: %d", len(body.text))

    language = detect_language(body.text)
    metrics = analyze_text(body.text, language)
    score = metrics["score"]
    logger.info("Score: %d, language: %s", score, language)

    suggestions = _get_fallback_suggestions(metrics, language, score)
    llm_available = False

    summary = _get_summary(score, language)

    await log_analysis(client_ip, language, score, len(body.text))

    response = AnalyzeResponse(
        score=score,
        detected_language=language,
        response_language=language,
        metrics=MetricsResponse(
            burstiness_score=metrics["burstiness_score"],
            lexical_diversity=metrics["lexical_diversity"],
            sentence_length_variance=metrics["sentence_length_variance"],
            repetition_score=metrics["repetition_score"],
            avg_sentence_length=metrics["avg_sentence_length"],
            unique_word_ratio=metrics["unique_word_ratio"],
        ),
        suggestions=[SuggestionResponse(**s) for s in suggestions],
        summary=summary,
        llm_available=llm_available,
    )

    logger.info("Returning analyze response, score=%d, suggestions=%d", score, len(suggestions))
    return JSONResponse(
        content=response.model_dump(),
        headers={"X-RateLimit-Remaining": str(remaining)},
    )


HUMANIZE_FALLBACK: dict[str, dict[str, str]] = {
    "it": {
        "error": "Il servizio di umanizzazione non e' disponibile al momento. Prova a seguire i consigli del rilevatore AI per migliorare il testo manualmente.",
    },
    "en": {
        "error": "The humanization service is not available right now. Try following the AI detector suggestions to improve the text manually.",
    },
    "es": {
        "error": "El servicio de humanizacion no esta disponible en este momento. Intenta seguir los consejos del detector de IA para mejorar el texto manualmente.",
    },
}


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize_endpoint(body: HumanizeRequest, request: Request) -> HumanizeResponse | JSONResponse:
    client_ip = _get_client_ip(request)

    allowed, remaining = await check_rate_limit(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again tomorrow."},
            headers={"X-RateLimit-Remaining": "0"},
        )

    language = detect_language(body.text)

    try:
        result, llm_available = await humanize_text(body.text, language)
    except Exception as e:
        logger.error("Humanize failed unexpectedly: %s", type(e).__name__)
        result, llm_available = None, False

    if not llm_available or result is None:
        lang = language if language in ("it", "en", "es") else "en"
        return JSONResponse(
            status_code=503,
            content={"error": HUMANIZE_FALLBACK[lang]["error"]},
            headers={"X-RateLimit-Remaining": str(remaining)},
        )

    await log_analysis(client_ip, language, 0, len(body.text))

    response = HumanizeResponse(
        original_text=body.text,
        humanized_text=result["humanized_text"],
        detected_language=language,
        changes_made=result["changes_made"],
        llm_available=True,
    )

    return JSONResponse(
        content=response.model_dump(),
        headers={"X-RateLimit-Remaining": str(remaining)},
    )
