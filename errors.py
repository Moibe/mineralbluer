"""Translate downloader/pipeline failures into something an end user can act on.

The raw yt-dlp text is written for someone at a terminal -- it is in English, names CLI flags,
and does not say whether waiting would help. The UI needs the opposite: plain language, and a
clear signal of whether retrying later is worthwhile.
"""

# (code, user-facing message, retryable) chosen by the first matching signature.
_SIGNATURES = [
    (
        ("429", "too many requests"),
        "rate_limited",
        "YouTube está limitando temporalmente las descargas desde esta red. "
        "No es un problema del video ni de tu enlace: suele resolverse solo en un rato. "
        "Intenta de nuevo más tarde.",
        True,
    ),
    (
        ("sign in to confirm you", "not a bot", "confirm you're not a bot"),
        "bot_check",
        "YouTube pidió una verificación antibot para este video. "
        "Suele ser temporal; intenta de nuevo en unos minutos.",
        True,
    ),
    (
        ("video unavailable", "video no disponible"),
        "unavailable",
        "El video no está disponible. Puede haber sido eliminado o tener restricción por región.",
        False,
    ),
    (
        ("private video",),
        "private",
        "El video es privado, así que no se puede analizar.",
        False,
    ),
    (
        ("members-only", "join this channel"),
        "members_only",
        "El video es exclusivo para miembros del canal, así que no se puede analizar.",
        False,
    ),
    (
        ("age-restricted", "age restricted", "inappropriate for some users"),
        "age_restricted",
        "El video tiene restricción de edad y no se puede descargar sin iniciar sesión.",
        False,
    ),
    (
        ("is not a valid url", "unsupported url", "incomplete youtube id"),
        "bad_url",
        "Ese enlace no parece ser un video de YouTube válido. Revisa que esté completo.",
        False,
    ),
    (
        ("requested format is not available",),
        "no_format",
        "YouTube no está ofreciendo una calidad utilizable para este video en este momento. "
        "Suele ser temporal; intenta de nuevo más tarde.",
        True,
    ),
    (
        ("unable to allocate", "memoryerror", "out of memory"),
        "out_of_memory",
        "El equipo se quedó sin memoria procesando este video. "
        "Prueba analizando un tramo más corto.",
        False,
    ),
    (
        ("timed out", "connection", "network is unreachable", "getaddrinfo"),
        "network",
        "Hubo un problema de conexión al descargar el video. Revisa tu internet e intenta de nuevo.",
        True,
    ),
    (
        ("rapidocr", "no module named 'rapidocr"),
        "ocr_missing",
        "Falta el motor de OCR. Instala las dependencias con "
        "`venv\\Scripts\\pip install -r requirements.txt`.",
        False,
    ),
]

_FALLBACK = (
    "unknown",
    "No se pudo completar el análisis de este video. "
    "Si vuelve a pasar, intenta con un tramo más corto u otro video.",
    False,
)


def classify(exc: Exception) -> dict:
    text = str(exc).lower()
    for needles, code, message, retryable in _SIGNATURES:
        if any(needle in text for needle in needles):
            return {"code": code, "message": message, "retryable": retryable}
    code, message, retryable = _FALLBACK
    return {"code": code, "message": message, "retryable": retryable}
