# Genre Classification

How fetched YouTube news is labeled in `generate_json.py`, and how those labels become feed tabs.

## Overview

Each video gets one genre. The label is chosen in three steps:

1. **Live / Scheduled** come from `videoType`, not from the title.
2. **Recent videos** (published in the last hour) try OpenAI, then a context check.
3. **Everything else**, and any OpenAI failure, uses keyword rules, then the same context check.

The context check is cosine similarity against short genre examples. Crime, road incidents, and other harm stories are all labeled **Negative**. The feed does not use Crime or Traffic as genres.

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
| 3 | Road harm (accidents, jams, collisions), checked before other harm so a highway death is claimed here | Negative |
| 4 | Jobs, unless the text is a scam, conversion story, or a school-exam human-interest clip | Jobs |
| 5 | Other harm (arrest, murder, raid, blast, fraud, and the Hindi equivalents) | Negative |
| 6 | Events, unless the text is a crime story or a political meet / rally / speech / election | Events |
| 7 | Politics, unless it is a fraud case or a launch / inauguration | Politics |
| 8 | Civic | Civic |
| 9 | Nothing matched | General |

## Context matching

After the keyword label, or after a successful OpenAI label, `semantic_context_scores` compares the video with example sentences for Negative, Politics, Events, Jobs, Civic, and General. Negative examples cover both crime and road incidents.

### Vector

The query is the title repeated three times plus the first 400 characters of the description.

Tokens:

- A word (unigram), weight 2. Words shorter than 2 characters are dropped. Stopwords such as `the`, `में`, `का`, `नहीं` are dropped.
- The next word (bigram), weight 3. A bigram is kept unless both words are on the unigram-skip list.
- A known harm phrase (`फसल बर्बाद`, `नहीं रहे`, `helicopter crash`, `शराब जब्त`, `गंभीर आरोप`, `आर्थिक संकट`, and the same list in code), weight 4.

Words that are too common to mean harm by themselves are not unigrams. They still count inside a phrase or a mixed bigram. That list includes `फसल`, `धान`, `बारिश`, `चक्रवात`, `तूफान`, `किसान`, `शराब`, and `helicopter`, so a weather line or an air show does not become Negative just because it mentions rain or a helicopter.

Each example sentence is its own document. Inverse document frequency is `log((1 + N) / (1 + df)) + 1`. Term frequency is `1 + log(count)`. The genre score is the **highest cosine** against that genre's examples, not an average, so one close example is enough.

### Decision

`apply_negative_context` then applies:

- Podcast, Live, Scheduled, and an existing Negative label stay as they are.
- A stored Crime or Traffic label is rewritten to Negative.
- Otherwise the label becomes Negative when the Negative cosine is at least `0.12` (`SEMANTIC_NEGATIVE_MIN`) and at least `0.72` times the best other genre (`SEMANTIC_NEGATIVE_RATIO`).

That is what moves crop loss, raids, deaths, blasts, fraud, missing-person stories, and road incidents into Negative when the keyword pass stopped on Politics, Events, Civic, or General because a minister or a city was also in the title.

Negative examples cover arrests, raids, fraud, violence, deaths, blasts, crop destruction, helicopter crashes, seizures, and road accidents, in English and Hindi. The other genres have their own short examples so a rally, a festival, a vacancy, a municipal notice, or a weather alert can outscore Negative.

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
| Negative | 7 | Genre Negative, newest `publishedAt` first |
| Education | 8 | Course lessons from the education channels only. Kept for six months. Not copied into other tabs or All. |

There is one Negative tab and one Negative rail. Each item's `genre` is `Negative`. Crime and Traffic are not written.

## OpenAI prompt

For the last-hour path, the prompt tells the model to choose Negative for crime, road accidents, and other harmful news, even when a politician is named. A reply of Crime or Traffic is stored as Negative. The cosine check still runs, so a harm story the model left in Politics or General can still move to Negative.
