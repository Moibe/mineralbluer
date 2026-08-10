# mineralbluer

Convierte un video de convención de [MineralBlu](https://www.youtube.com/@MineralBlu) en un catálogo
estructurado: **foto + personaje + serie + twitter** de cada cosplayer.

El canal rotula a cada cosplayer con un bloque de texto quemado en el video. Los videos viejos traen
un handle pelón; los nuevos nombran la plataforma y listan varias cuentas:

```
UBEL                    SUE STORM
FRIEREN                 MARVEL RIVALS
@tinechan               twitch @alinity
                        ig @alinity_divine
```

Este backend descarga el video, lo recorre ~1 frame por segundo leyendo ese bloque con OCR, agrupa
los frames consecutivos que muestran el mismo rótulo (una persona = un grupo) y guarda las mejores
fotos de cada grupo junto con los datos leídos.

- **Backend (este repo)**: Python + FastAPI. Es el dueño de la base y de las imágenes.
- **Frontend**: [`../mineralbluer-front`](../mineralbluer-front) — SvelteKit 5 + Tailwind v4. Sólo
  consume la API.

## Correr

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

Y en otra terminal, el front:

```bash
cd ../mineralbluer-front
npm install
npm run dev     # http://localhost:7979
```

La primera corrida descarga los modelos de PP-OCR (~15 MB) e InsightFace (~300 MB) a `~/.cache`.

### Cookies de YouTube (importante)

Sin cookies, YouTube responde *"Sign in to confirm you're not a bot"* al cliente por defecto y
`yt-dlp` cae a clientes alternativos. Medido en este proyecto:

| Cliente | Resolución | Velocidad |
|---|---|---|
| default (con cookies) | 1440p+ | ~1.3 MB/s |
| `web_embedded` | 1440p | **~32 KB/s** — funciona pero es 40x más lento |
| `android`/`tv` | 360p | ~1.3 MB/s |

A 360p el rótulo mide ~13 px de alto y el OCR devuelve basura, así que la cadena prefiere
`web_embedded` (lento pero legible) antes que caer a 360p. **Para uso real, pon cookies** y todo
sale rápido y en alta:

```bash
# Opción A: leer del navegador. Falla si Chrome está abierto (bloquea su base de cookies).
set MINERALBLUER_COOKIES_FROM_BROWSER=chrome

# Opción B: exportar cookies.txt una vez (con cualquier extensión "Get cookies.txt") y apuntar aquí.
set MINERALBLUER_COOKIES_FILE=C:\ruta\cookies.txt
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `main.py` | API FastAPI: jobs de análisis + CRUD del catálogo + `/media` estático. |
| `jobs.py` | Análisis en hilo aparte, con progreso, cancelación y recuperación tras reinicio. |
| `pipeline.py` | Descarga → muestreo de frames → segmentación → elección de fotos → guardado. |
| `overlay.py` | **El corazón**: OCR del rótulo, parseo de las 3 líneas y votación entre frames. |
| `db.py` | El catálogo en SQLite. |
| `config.py` | Rutas de `storage/`. |
| `errors.py` | Traduce fallos de yt-dlp a algo accionable en español. |
| `scripts/run_pipeline.py` | Correr un análisis completo desde la terminal. |
| `scripts/probar_ocr.py` | Volcar frame por frame qué leyó el OCR (para afinar `overlay.py`). |

Todo lo generado vive en `storage/` (ignorado por git): `catalog.db`, `media/<video>/<seg>/*.jpg`,
`video_cache/`, `jobs/`.

## Cómo se decide qué es un rótulo

El ancla es la **geometría de la pila**: dos o más líneas que comparten borde izquierdo y tamaño de
letra, apiladas una debajo de otra. Un video de convención está lleno de texto incidental (lonas,
gafetes, señalética) pero casi nunca forma una columna así de limpia.

Una versión anterior anclaba en la `@` y perdía más de la mitad de los rótulos: **PP-OCR se come esa
glifo constantemente** y devuelve `calebweekss` o `wanderlustluca` sin arroba. Medido sobre 50
frames del mismo clip, el cambio de ancla pasó de 11 a 33 rótulos detectados.

Otras decisiones que salieron de medir, no de suponer:

- **No se recorta el frame.** La posición vertical del rótulo NO es estable: en 50 segundos del
  mismo video aparecieron rótulos que empiezan entre el 22% y el 51% de la altura. Cualquier recorte
  probado le cortaba la primera línea a alguno — es decir, el nombre del personaje.
- Las cajas de una misma fila se **reunen** antes de analizar: `twitch @alinity` vuelve del OCR como
  dos cajas, y separadas una parece una palabra suelta y la otra pierde su plataforma.
- La tolerancia de alineación es de **2 alturas de línea**, no menos: cuando el OCR pierde la `@`
  inicial, esa línea empieza un caracter más a la derecha (72 px contra un presupuesto de 71 px era
  suficiente para romper la pila).
- Dentro de la pila el reparto es **por contenido, no por posición**: las líneas con forma de cuenta
  (una palabra, sin espacios, mayormente minúsculas) van a `handles`; el resto es personaje (la
  primera) y serie (las demás). Leerlo por posición archivaba `twitch @alinity` como parte de la
  serie.
- El logo **MINERALBLU** se filtra con comparación difusa: el OCR lo lee distinto en cada frame
  (`MINERALBLL`, `MINGRALBLO`, `MINERALBLD`…) y un match exacto deja pasar casi todas.
- Cada segmento **vota** entre todos sus frames en vez de creerle a una sola lectura. El desacuerdo
  se guarda como `confidence`, y el front usa eso para marcar qué vale la pena revisar.
- Un rótulo visto en **un solo frame** se descarta (`MIN_READINGS = 2`): casi siempre es ruido.

### Twitter vs. las demás cuentas

Los videos viejos rotulan con un `@handle` pelón; los nuevos nombran la plataforma y suelen listar
dos o tres (`twitch @alinity`, `ig @alinity_divine`). Se guardan **todas** en `handles`, y la columna
`twitter` se llena sólo cuando el rótulo realmente traía un twitter/x o un handle pelón. Cuando no
hay twitter queda en `NULL` en vez de pasar un instagram por uno — el front muestra las otras
cuentas igual, así que no se pierde nada por ser exacto.

## Editar a mano, y qué pasa al re-analizar

Volver a analizar un video **reemplaza** sus entradas, con dos excepciones deliberadas: las que
editaste (`verified`) y las que descartaste (`deleted`) sobreviven, y cualquier segmento nuevo que
caiga sobre ese mismo tramo se ignora. Sin eso, un segundo análisis (para probar 4K, o para ampliar
el tramo) borraría tus correcciones y resucitaría lo que ya habías tirado.

## API

| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/jobs` | Lanza un análisis. `{url, start_seconds?, end_seconds?, quality?, force_redownload?}` |
| `GET` | `/jobs` · `/jobs/history` · `/jobs/{id}` | Activos · historial · estado y resultado |
| `POST` | `/jobs/{id}/cancel` | Cancela |
| `GET` | `/cosplays` | Catálogo. Filtros: `q`, `series`, `video_id`, `verified`, `order`, `limit`, `offset` |
| `PATCH` | `/cosplays/{id}` | Corrige personaje/serie/twitter, o promueve otro frame a foto. Marca `verified` |
| `DELETE` | `/cosplays/{id}` | Descarta (soft). `?hard=true` borra la fila de verdad |
| `GET` | `/series` · `/videos` · `/stats` | Facetas para los filtros |
| `GET` | `/media/...` | Las fotos |

## Límites conocidos

- **Costo**: el OCR domina el tiempo. Medido: **~3.4 s por frame muestreado** (PP-OCR en CPU, video
  1440p, ancho de OCR 1280). A 1 frame/s eso es ~3.4 min por cada minuto de video, así que un video
  de 20 min ronda la hora. `MAX_SAMPLES = 1200` acota cualquier corrida a ~70 min bajando el muestreo
  en videos largos; para probar algo, analiza un tramo.
- **Recall**: sobre 50 frames de prueba se reconoció el rótulo en 35, y salieron **8 de 9**
  cosplayers del tramo, sin un solo falso positivo. El que se perdió tenía el handle tan mal leído
  (`ntinel`, `ifin`, `as` en frames distintos) que ninguna lectura era recuperable.
- **Rótulo sin ninguna línea con forma de cuenta**: no se detecta. Exigir eso es lo que mantiene los
  falsos positivos en cero; sin ese requisito, cualquier par de líneas de una lona entra al catálogo.
- **Menos de 720p**: el OCR devuelve texto plausible pero equivocado, que es peor que no devolver
  nada. El análisis avisa con un warning en vez de fingir que salió bien.
