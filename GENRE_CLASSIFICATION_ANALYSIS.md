# Genre Classification Analysis Report

## Overview

Analysis of the genre classification system in `generate_json.py`. The system uses a **hybrid approach**: OpenAI API for recently published videos (last 1 hour) and rule-based keyword matching for older videos and fallback.

## Current Implementation

### Classification Strategy

1. **Special types (no ML/keywords)**  
   - **Live**: `videoType === "LIVE"` → genre `Live`  
   - **Scheduled**: `videoType` contains `"SCHEDULED"` or `"UPCOMING"` → genre `Scheduled`

2. **Recent videos (published in last 1 hour)**  
   - Try **OpenAI** classification first (see [OpenAI Genre Classification Integration](OPENAI_CLASSIFICATION.md)).  
   - **Circuit breaker**: On first OpenAI failure (error, rate limit, timeout), stop using OpenAI for the rest of the batch and use keyword-based only.  
   - **Time limit**: Stop using OpenAI after `OPENAI_MAX_EXECUTION_TIME` (50s); remaining videos use keyword-based.  
   - **Podcast override**:  
     - If OpenAI returns Podcast but channel is not in `PODCAST_ALLOWED_CHANNEL_TITLES` → reclassify with keyword-based.  
     - If OpenAI returns non-Podcast but channel is in `PODCAST_ALLOWED_CHANNEL_TITLES` → force genre `Podcast`.

3. **Older videos (or when OpenAI is skipped)**  
   - **Keyword-based** only, with `channel_title` used for Podcast.

### Genres

| Genre     | Constant         | Notes |
|----------|-------------------|-------|
| Live     | `GENRE_LIVE`      | Currently live stream. |
| Scheduled| `GENRE_SCHEDULED` | Upcoming/scheduled stream. |
| Crime    | `GENRE_CRIME`     | Crime, police, courts, violence, etc. |
| Traffic  | `GENRE_TRAFFIC`   | Accidents, jams, road incidents. |
| Jobs     | `GENRE_JOBS`      | Recruitment, vacancies, exams, interviews. |
| Events   | `GENRE_EVENTS`    | Festivals, ceremonies, inaugurations. |
| Civic    | `GENRE_CIVIC`     | Municipal services, civic issues, utilities. |
| Politics | `GENRE_POLITICS`  | Elections, rallies, government, CM/PM. |
| Podcast  | `GENRE_PODCAST`   | From allowed podcast channels only. |
| General  | `GENRE_GENERAL`   | Default / weather / everything else. |

Podcast is restricted to channels in `PODCAST_ALLOWED_CHANNEL_TITLES` (e.g. "Chhattisgarh Podcast", "Z-Series CG Podcast", "The Lok Ras", "The PS Show"). Videos from these channels are always classified as Podcast in the feed.

---

## Keyword-Based Algorithm

Used by `classify_genre_keyword_based(title, description, channel_title)` when OpenAI is not used or fails.

### 1. Channel-based Podcast (first)

- If `channel_title in PODCAST_ALLOWED_CHANNEL_TITLES` → return **Podcast** immediately.

### 2. Text normalization

- Lowercase; remove special characters (keep alphanumeric + Devanagari).
- Normalize spacing and apply fixed misspelling replacements (e.g. `h!dsa` → `हादसा`, `mou.t` → `मौत`).

### 3. Title weighting

- Search text: `title + title + title + description` (title effectively 3× weight).

### 4. Pattern matching

- Regex with word boundaries for English; space boundaries for Hindi/Devanagari.
- Categories are checked in this order (first match wins):

| Order | Category | Notes |
|-------|----------|--------|
| 1 | **Weather** | Match → return **General** (avoids "MP Weather" matching Politics). |
| 2 | **Traffic** | Before Crime so road accidents with deaths stay Traffic. |
| 3 | **Interview** | Podcast-like keywords from non–podcast channels; job-interview cases skip to Jobs. |
| 4 | **Jobs** | Recruitment, vacancy, exam, interview, etc. |
| 5 | **Crime** | Crime/police/court/violence terms. |
| 6 | **Events** | With exclusion for clear political terms (e.g. meets, रैली, भाषण, चुनाव). |
| 7 | **Politics** | With exclusions for scam/fraud (crime) and launch/inauguration (events). |
| 8 | **Civic** | Municipal, civic, certificates, utilities. |
| 9 | **General** | Default. |

### 5. Keyword lists

- Each genre has curated English and Hindi/Devanagari keywords and multi-word phrases (e.g. "traffic accident", "सड़क दुर्घटना", "police recruitment", "नौकरी सूचना").
- Context rules: e.g. job-interview indicators prevent Interview from winning over Jobs; crime/event exclusions refine Politics vs Crime/Events.

---

## Accuracy Assessment

### Overall

- **Keyword-based (standalone)**: ~70–80% (unchanged from previous report).
- **With OpenAI for recent videos**: Recent items can reach ~85–90% when OpenAI is used successfully; older items remain at keyword-based accuracy.
- Actual accuracy depends on rate limits, time limit, and circuit breaker (how often OpenAI is used in a run).

### Strengths

1. **Hybrid strategy**: Better accuracy for new content (OpenAI), predictable fallback (keywords).
2. **Title weighting**: Titles weighted 3× in keyword path; aligns with typical news titles.
3. **Word-boundary matching**: Reduces partial matches (e.g. "job" vs "jobless").
4. **Multi-language**: English and Hindi/Devanagari supported.
5. **Ordered categories**: Specific genres (e.g. Traffic, Crime, Jobs) checked before broader ones (Politics, General).
6. **Podcast by channel**: Clean separation for dedicated podcast channels.
7. **Context rules**: Exclusions for job interviews, political events, scam vs crime, etc., reduce obvious mislabels.
8. **Resilience**: Circuit breaker and time limit avoid runaway latency or cost; script never depends on OpenAI alone.

### Weaknesses and limitations

#### 1. False positives (keyword path)

- Generic terms can match in wrong contexts: "traffic" in "political traffic", "event" in "news event", "job" in "job well done".
- **Impact**: ~15–20% false positive rate for keyword-only classification.

#### 2. False negatives (keyword path)

- Missing terms or paraphrases: "auto crash" vs "car accident", "employment news" without "job", new slang/abbreviations.
- **Impact**: ~10–15% false negative rate for keyword-only.

#### 3. Ambiguous / multi-category

- e.g. "Police investigate traffic accident" (Crime vs Traffic), "Government announces new job scheme" (Politics vs Jobs).  
- **Behavior**: First matching category wins; no confidence or multi-label.

#### 4. No semantic understanding (keyword path)

- Purely lexical; e.g. "Dead body found" can match crime keywords even for natural death; "party" can be event vs politics.

#### 5. Keyword coverage gaps

- Traffic: e.g. "collision", "pile-up", "fender bender".
- Jobs: e.g. "opening", "position", "career opportunity", "hiring drive".
- Events: e.g. "gathering", "function", "ceremony", "conference".
- Civic: some regional/service terms may be missing.

#### 6. No negative keywords

- Cannot explicitly exclude e.g. "traffic light" (discussion) or "job interview" (story, not posting) except via context rules.

#### 7. No scoring

- First match wins; no confidence score or strength of signal.

#### 8. Description quality

- Relies on YouTube title/description: can be empty, auto-generated, SEO spam, or non–Hindi/English.

---

## Accuracy by Genre (keyword-based)

| Genre    | Est. accuracy | Common issues |
|----------|----------------|----------------|
| Crime    | ~85%           | Non-crime police activities may be labeled Crime. |
| Traffic  | ~75%           | Overlap with civic "road repair" or general traffic discussion. |
| Jobs     | ~70%           | "Job" is generic; non-job content can match. |
| Events   | ~75%           | "Event" is generic; overlaps with general news. |
| Civic    | ~65%           | Broad; overlaps with Politics, Events, Traffic. |
| Politics | ~70%           | "Government", "party" overlap with Civic and General. |
| Podcast  | ~100% (channel)| By channel only; no keyword confusion. |
| General  | ~80%           | Default; correctly catches unclassified. |

---

## Real-world factors

**Positive**

- News titles are often descriptive and keyword-rich.
- Hindi news often uses consistent terminology.
- Title weighting and category order improve relevance.
- Word-boundary and context rules reduce many false hits.

**Negative**

- Clickbait or vague headlines ("Breaking: Major Incident in City").
- Mixed language and transliterations.
- Abbreviations and slang not in keyword lists.
- Domain-specific phrasing not covered.

---

## Recommendations for improvement

### 1. Context-aware matching (high impact)

- Require supporting keywords for generic terms (e.g. "job" + recruitment/vacancy/hiring).
- Reduces false positives for Jobs/Events/Traffic.

### 2. Scoring system (high impact)

- Score by: keyword specificity, number of matches, title vs description, multi-word phrase matches.
- Use threshold to leave low-confidence as General.

### 3. Negative keywords (medium impact)

- e.g. "traffic" + "light" → not Traffic incident; "job" + "story" → not Jobs posting.

### 4. Expand keyword lists (medium impact)

- Synonyms, misspellings, regional terms, abbreviations (e.g. BMC, NMC for Civic).

### 5. Flexible multi-word matching (medium impact)

- Allow "road accident" ≈ "accident on road"; "traffic jam" ≈ "traffic congestion".

### 6. ML/NLP model (high impact, high effort)

- Train on labeled examples; TF-IDF or embeddings; target 85–90% on full feed.

### 7. Confidence thresholds (low effort)

- In keyword path: only assign non-General if confidence > threshold; else General.

---

## Expected accuracy after improvements

| Change                     | Expected gain (keyword path) |
|----------------------------|------------------------------|
| Context-aware matching     | +5–8%                        |
| Scoring system             | +8–12%                       |
| Negative keywords          | +3–5%                        |
| Expanded keyword lists     | +2–4%                        |
| ML-based approach          | +10–15%                      |

**Potential ceiling**: ~85–90% for keyword-based path with all improvements; recent-video path already benefits from OpenAI when available.

---

## Conclusion

The current system is a **hybrid**: OpenAI for videos from the last hour (with circuit breaker and time limit), and **keyword-based** for everything else and fallback. It adds **Live**, **Scheduled**, and **Podcast** (channel-gated) and keeps the previous keyword design (order, context rules, normalization).

- **Works well for**: Clear, keyword-rich titles; standard news terminology; single-category content; dedicated podcast channels.
- **Struggles with**: Ambiguous or multi-category content; context-dependent meanings; generic keyword overlap; missing domain terms.

For production, the existing circuit breaker and time limit are already in place. Quick wins to improve keyword accuracy further: **scoring** and **context-aware matching**, which could bring the keyword-only share to ~75–85% with moderate effort. See [OpenAI Genre Classification Integration](OPENAI_CLASSIFICATION.md) for OpenAI setup and behavior.
