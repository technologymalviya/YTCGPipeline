# Genre Classification

How fetched YouTube news is labeled in `generate_json.py`, and how those labels become feed tabs.

## Overview

Each video gets one genre. The label is chosen in three steps:

1. **Live / Scheduled** come from `videoType`, not from the title.
2. **Recent videos** (published in the last hour) try OpenAI, then a context check.
3. **Everything else**, and any OpenAI failure, uses keyword rules, then the same context check.

The context check is cosine similarity against short genre examples. It pulls negative stories into **Crime** when the headline is about harm even if a politician, city, or scheme is also mentioned. Road stories stay **Traffic**.

Crime and Traffic are still separate genres. The feed shows them as one tab, **Negative**.

## When each path runs

`add_genres_to_feed` walks the fetched list:

| Video | Genre |
| --- | --- |
| `videoType` is `LIVE` | `Live` |
| `videoType` contains `SCHEDULED` or `UPCOMING` | `Scheduled` |
| Published in the last hour, and OpenAI is available | OpenAI label, then context check |
| Older, or OpenAI failed, rate-limited, or out of time | Keywords, then context check |

OpenAI stops for the rest of the batch after the first failure, and after `OPENAI_MAX_EXECUTION_TIME` (50 seconds). Those remaining videos use keywords.

Scheduled videos keep the genre `Scheduled` and are left out of every tab. There is no Scheduled row.

## Podcast

A video is **Podcast** before keywords or the context check when either of these is true:

- The channel is a podcast channel: `Chhattisgarh Podcast`, `Z-Series CG Podcast`, `The Lok Ras`, `The PS Show`, or any channel whose name contains `podcast` / `पॉडकास्ट` / `पोडकास्ट`. Every upload from that channel goes in the Podcast row.
- The title or description itself says podcast (`podcast`, `पॉडकास्ट`, `z-series`, and the same family of spellings).

OpenAI has no Podcast label. After OpenAI returns, the same channel and title rules still force Podcast.

## Keyword pass

`_classify_genre_by_keywords` normalizes the title and description (lowercase, punctuation stripped, a few misspellings rewritten) and searches `title` three times plus the description.

English keywords use word boundaries. Hindi keywords match on spaces. The first matching category wins:

| Order | Match | Genre |
| --- | --- | --- |
| 1 | Weather | General |
| 2 | Health outbreak (jaundice, dengue, and similar) | General |
| 3 | Traffic (accidents, jams, collisions). Checked before Crime so a death on the road stays Traffic. | Traffic |
| 4 | Jobs, unless the text is a scam, conversion story, or a school-exam human-interest clip | Jobs |
| 5 | Crime keywords (arrest, murder, raid, blast, fraud, and the Hindi equivalents) | Crime |
| 6 | Events, unless the text is a crime story or a political meet / rally / speech / election | Events |
| 7 | Politics, unless it is a fraud case or a launch / inauguration | Politics |
| 8 | Civic | Civic |
| 9 | Nothing matched | General |

## Context matching

After the keyword label, or after a successful OpenAI label, `semantic_context_scores` compares the video with example sentences for Crime, Traffic, Politics, Events, Jobs, Civic, and General.

### Vector

The query is the title repeated three times plus the first 400 characters of the description.

Tokens:

- A word (unigram), weight 2. Words shorter than 2 characters are dropped. Stopwords such as `the`, `में`, `का`, `नहीं` are dropped.
- The next word (bigram), weight 3. A bigram is kept unless both words are on the unigram-skip list.
- A known harm phrase (`फसल बर्बाद`, `नहीं रहे`, `helicopter crash`, `शराब जब्त`, `गंभीर आरोप`, `आर्थिक संकट`, and the same list in code), weight 4.

Words that are too common to mean harm by themselves are not unigrams. They still count inside a phrase or a mixed bigram. That list includes `फसल`, `धान`, `बारिश`, `चक्रवात`, `तूफान`, `किसान`, `शराब`, and `helicopter`, so a weather line or an air show does not become Crime just because it mentions rain or a helicopter.

Each example sentence is its own document. Inverse document frequency is `log((1 + N) / (1 + df)) + 1`. Term frequency is `1 + log(count)`. The genre score is the **highest cosine** against that genre's examples, not an average, so one close example is enough.

### Decision

`apply_negative_crime_context` then applies:

- Podcast, Live, Scheduled, and an existing Crime label stay as they are.
- Traffic stays Traffic. Accidents are not moved into Crime.
- Otherwise the label becomes Crime when the Crime cosine is at least `0.12` (`SEMANTIC_CRIME_MIN`) and at least `0.72` times the best other genre (`SEMANTIC_CRIME_RATIO`).

That is what moves crop loss, raids, deaths, blasts, fraud, and missing-person stories into Crime when the keyword pass stopped on Politics, Events, Civic, or General because a minister or a city was also in the title.

Crime examples cover arrests, raids, fraud, violence, deaths, blasts, crop destruction, helicopter crashes, and seizures, in English and Hindi. The other genres have their own short examples so a rally, a festival, a vacancy, a municipal notice, or a weather alert can outscore Crime.

## Feed tabs

Genres are grouped into sections in `generate_ott_json`. `sectionIndex` is the rail order.

| Section | Index | What is in it |
| --- | --- | --- |
| City (when an IP resolves) | 0 | Videos whose title names that city. This is extra; the same video also sits in its genre tab. |
| Live | 0 | Genre Live |
| General | 1 | Genre General |
| Podcast | 2 | Genre Podcast |
| Jobs | 3 | Genre Jobs |
| Politics | 4 | Genre Politics |
| Events | 5 | Genre Events |
| Civic | 6 | Genre Civic |
| Negative | 7 | Genre Crime and genre Traffic, newest `publishedAt` first |

There is one Negative tab and one Negative rail. Individual items still carry `genre` `Crime` or `Traffic`.

## OpenAI prompt

For the last-hour path, the prompt tells the model that Crime includes negative news (deaths, disasters, crop loss, blasts, raids, arrests, fraud, missing people, violence, allegations), even when a politician is named, and that Traffic is only road accidents, jams, and collisions. The cosine check still runs on that label, so a non-traffic story the model left in Politics or General can still move to Crime.
