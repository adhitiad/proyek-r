import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from groq import Groq
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.graph import END, StateGraph


try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except Exception as exc:  # pragma: no cover - runtime import guard
    webdriver = None
    By = None
    Keys = None
    WebDriverWait = None
    EC = None
    _SELENIUM_IMPORT_ERROR = exc
else:
    _SELENIUM_IMPORT_ERROR = None


@dataclass
class SourceConfig:
    name: str
    search_url: Optional[str] = None
    results_selector: Optional[str] = None
    title_selector: Optional[str] = None
    date_selector: Optional[str] = None
    snippet_selector: Optional[str] = None
    base_url: Optional[str] = None
    use_ui: bool = False
    ui_url: Optional[str] = None
    ui_input_selectors: Optional[List[str]] = None
    ui_submit_selectors: Optional[List[str]] = None


def _log(message: str) -> None:
    print(f"[news-agent] {message}", file=sys.stderr)


def load_env() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "groq/compound"),
    }


def build_driver(headless: bool) -> "webdriver.Chrome":
    if webdriver is None:
        raise RuntimeError(f"Selenium import failed: {_SELENIUM_IMPORT_ERROR}")

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def search_via_ui(driver: "webdriver.Chrome", config: SourceConfig, query: str) -> bool:
    if not config.ui_url or not config.ui_input_selectors:
        return False

    driver.get(config.ui_url)
    time.sleep(2)

    input_el = None
    for selector in config.ui_input_selectors:
        try:
            input_el = driver.find_element(By.CSS_SELECTOR, selector)
            if input_el:
                break
        except Exception:
            continue

    if not input_el:
        return False

    try:
        input_el.clear()
        input_el.send_keys(query)
        input_el.send_keys(Keys.ENTER)
    except Exception:
        return False

    if config.results_selector:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, config.results_selector))
            )
        except Exception:
            pass
    else:
        time.sleep(3)

    return True


def fetch_search_page(driver: "webdriver.Chrome", config: SourceConfig, query: str) -> Optional[str]:
    if config.use_ui:
        ok = search_via_ui(driver, config, query)
        if ok:
            return driver.page_source

    if config.search_url:
        url = config.search_url.format(query=quote_plus(query))
        driver.get(url)
        time.sleep(2)
        return driver.page_source

    return None


def extract_articles_from_item(
    item,
    config: SourceConfig,
    default_source: str,
) -> Optional[Dict]:
    title_el = item.select_one(config.title_selector) if config.title_selector else None
    if not title_el:
        return None

    title = title_el.get_text(" ", strip=True)
    url = title_el.get("href", "").strip()
    if config.base_url:
        url = urljoin(config.base_url, url)

    if not title or not url:
        return None

    date_text = ""
    if config.date_selector:
        date_el = item.select_one(config.date_selector)
        if date_el:
            date_text = date_el.get_text(" ", strip=True)

    snippet_text = ""
    if config.snippet_selector:
        snippet_el = item.select_one(config.snippet_selector)
        if snippet_el:
            snippet_text = snippet_el.get_text(" ", strip=True)

    return {
        "title": title,
        "url": url,
        "date": date_text,
        "snippet": snippet_text,
        "source": default_source,
    }


def extract_articles_generic(
    soup: BeautifulSoup,
    source: str,
    base_url: Optional[str],
    domain_hint: Optional[str],
) -> List[Dict]:
    articles: List[Dict] = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        if domain_hint:
            if domain_hint not in href and not href.startswith("/"):
                continue

        title = anchor.get_text(" ", strip=True)
        if len(title) < 20:
            continue

        url = urljoin(base_url or "", href)
        if url in seen:
            continue

        parent = anchor.find_parent(["article", "div", "li"])
        snippet = ""
        date_text = ""
        if parent:
            p = parent.find("p")
            if p:
                snippet = p.get_text(" ", strip=True)
            time_tag = parent.find("time")
            if time_tag:
                date_text = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)

        articles.append(
            {
                "title": title,
                "url": url,
                "date": date_text,
                "snippet": snippet,
                "source": source,
            }
        )
        seen.add(url)

    return articles


def parse_search_results(html: str, config: SourceConfig) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    if config.results_selector and config.title_selector:
        results = []
        for item in soup.select(config.results_selector):
            article = extract_articles_from_item(item, config, config.name)
            if article:
                results.append(article)
        return results

    domain_hint = None
    if config.base_url:
        domain_hint = config.base_url.replace("https://", "").replace("http://", "")

    return extract_articles_generic(soup, config.name, config.base_url, domain_hint)


def fetch_detail_text(url: str, driver: Optional["webdriver.Chrome"], use_driver: bool) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    text = ""

    if not use_driver:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = " ".join([p for p in paragraphs if len(p) > 40])
                return text.strip()
        except Exception:
            pass

    if driver is None:
        return text

    try:
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if len(p) > 40])
    except Exception:
        return ""

    return text.strip()


class AnalysisState(TypedDict, total=False):
    text: str
    api_key: str
    model: str
    prompt: str
    response_text: str
    analysis: Dict[str, Any]


_PROMPT_TEMPLATE = HumanMessage(
    input_variables=["text"],
    template=(
        "Analisis sentimen berita berikut untuk pasar saham Indonesia. "
        "Jawab dengan JSON saja, format:\\n"
        "{\\n"
        "  \\\"sentiment\\\": \\\"positive/negative/neutral\\\",\\n"
        "  \\\"confidence\\\": 0-1,\\n"
        "  \\\"impact_score\\\": 0-1,\\n"
        "  \\\"key_points\\\": [\\\"...\\\"],\\n"
        "  \\\"summary\\\": \\\"...\\\",\\n"
        "  \\\"suggested_action\\\": \\\"buy/hold/sell\\\"\\n"
        "}\\n\\n"
        "Berita: \"{text}\""
    ),
)

_ANALYSIS_GRAPH = None


def _default_analysis() -> Dict[str, Any]:
    return {
        "sentiment": "neutral",
        "confidence": 0.5,
        "impact_score": 0.5,
        "key_points": [],
        "summary": "",
        "suggested_action": "hold",
    }


def _node_build_prompt(state: AnalysisState) -> Dict[str, Any]:
    prompt = _PROMPT_TEMPLATE
    return {"prompt": prompt}


def _node_call_groq(state: Any) -> Dict[str, Any]:
    api_key = state.get("api_key", "")
    if not api_key:
        return {"analysis": _default_analysis()}

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=450,
            messages=[
                {
                    "role": "user",
                    "content": state["prompt"]
                }
            ]
        )
        if completion.choices:
            return {"analysis": completion.choices[0].message.content}
        data = completion.choices[0].message.content
        return {"response_text": data}
    except Exception:
        return {"analysis": _default_analysis()}


def _node_parse_response(state: AnalysisState) -> Dict[str, Any]:
    if state.get("analysis"):
        return {}
    content = state.get("response_text", "")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return {"analysis": json.loads(match.group())}
        except Exception:
            return {"analysis": _default_analysis()}
    return {"analysis": _default_analysis()}


def _get_graph():
    global _ANALYSIS_GRAPH
    if _ANALYSIS_GRAPH is not None:
        return _ANALYSIS_GRAPH
    builder = StateGraph(AnalysisState)
    builder.add_node("build_prompt", _node_build_prompt)
    builder.add_node("call_groq", _node_call_groq)
    builder.add_node("parse_response", _node_parse_response)
    builder.set_entry_point("build_prompt")
    builder.add_edge("build_prompt", "call_groq")
    builder.add_edge("call_groq", "parse_response")
    builder.add_edge("parse_response", END)
    _ANALYSIS_GRAPH = builder.compile()
    return _ANALYSIS_GRAPH


def groq_analyze(text: str, api_key: str, model: str) -> Dict[str, Any]:
    graph = _get_graph()
    result = graph.invoke({"text": text, "api_key": api_key, "model": model})
    return result.get("analysis") or _default_analysis()


def aggregate_results(articles: List[Dict]) -> Dict:
    if not articles:
        return {
            "avg_score": 0,
            "confidence": 0,
            "sentiment": "neutral",
            "news_count": 0,
        }

    total_weight = 0.0
    weighted_score = 0.0
    confidences = []

    for article in articles:
        analysis = article.get("analysis", {})
        sentiment = analysis.get("sentiment", "neutral")
        confidence = float(analysis.get("confidence", 0.5))
        score = 1 if sentiment == "positive" else -1 if sentiment == "negative" else 0
        weighted_score += score * confidence
        total_weight += confidence
        confidences.append(confidence)

    avg_score = weighted_score / total_weight if total_weight > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    if avg_score > 0.2:
        sentiment_label = "positive"
    elif avg_score < -0.2:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    return {
        "avg_score": avg_score,
        "confidence": avg_conf,
        "sentiment": sentiment_label,
        "news_count": len(articles),
    }


def build_sources() -> List[SourceConfig]:
    return [
        SourceConfig(
            name="detik",
            search_url="https://www.detik.com/search/searchall?query={query}",
            results_selector=".media",
            title_selector=".media__title a",
            date_selector=".media__date",
            base_url="https://www.detik.com",
        ),
        SourceConfig(
            name="tempo",
            search_url="https://www.tempo.co/search?q={query}",
            results_selector=".article-list-item",
            title_selector="h2 a",
            date_selector=".date",
            base_url="https://www.tempo.co",
        ),
        SourceConfig(
            name="cnbc",
            search_url="https://www.cnbcindonesia.com/search?q={query}",
            results_selector=".search-list-item",
            title_selector="h3 a",
            date_selector=".date",
            base_url="https://www.cnbcindonesia.com",
        ),
        SourceConfig(
            name="kontan",
            search_url="https://investasi.kontan.co.id/search?q={query}",
            results_selector=".article-item",
            title_selector="h3 a",
            date_selector=".date",
            base_url="https://investasi.kontan.co.id",
        ),
        SourceConfig(
            name="bisnis",
            search_url="https://market.bisnis.com/search?q={query}",
            results_selector=".article-item",
            title_selector="h3 a",
            date_selector=".date",
            base_url="https://market.bisnis.com",
        ),
        SourceConfig(
            name="investing",
            search_url="https://www.investing.com/search/?q={query}",
            use_ui=True,
            ui_url="https://www.investing.com",
            ui_input_selectors=[
                "input[name='q']",
                "input[placeholder*='Search']",
                "input[placeholder*='search']",
            ],
            base_url="https://www.investing.com",
        ),
        SourceConfig(
            name="metrotvnews",
            use_ui=True,
            ui_url="https://www.metrotvnews.com",
            ui_input_selectors=[
                "input[type='search']",
                "input[name='q']",
                "input[name='search']",
                "input[placeholder*='Cari']",
                "input[placeholder*='cari']",
            ],
            base_url="https://www.metrotvnews.com",
        ),
    ]


def normalize_query(symbol: Optional[str], query: Optional[str]) -> str:
    if query:
        return query.strip()
    if not symbol:
        return ""
    return symbol.replace(".JK", "").strip()


def run_agent(args: argparse.Namespace) -> Dict:
    env = load_env()
    api_key = env["GROQ_API_KEY"]
    model = env["GROQ_MODEL"] or "groq/compound"

    query = normalize_query(args.symbol, args.query)
    if not query:
        raise ValueError("Either --symbol or --query is required")

    sources = build_sources()
    per_source_limit = max(3, args.max_articles // max(1, len(sources)))

    driver = build_driver(args.headless)
    all_articles: List[Dict] = []

    try:
        for config in sources:
            _log(f"Searching {config.name} for '{query}'")
            html = fetch_search_page(driver, config, query)
            if not html:
                _log(f"No HTML returned for {config.name}")
                continue

            articles = parse_search_results(html, config)
            if not articles:
                _log(f"No articles parsed for {config.name}")
                continue

            all_articles.extend(articles[:per_source_limit])
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    # Deduplicate by URL
    deduped: List[Dict] = []
    seen = set()
    for article in all_articles:
        url = article.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(article)

    deduped = deduped[: args.max_articles]

    # Hybrid detail fetch
    detail_limit = max(0, args.detail_limit)
    detail_count = 0

    driver = None
    if detail_limit > 0:
        driver = build_driver(args.headless)

    try:
        for idx, article in enumerate(deduped):
            snippet = article.get("snippet", "") or ""
            need_detail = len(snippet) < 40 or idx < detail_limit
            full_text = ""
            if need_detail:
                detail_count += 1
                full_text = fetch_detail_text(article["url"], driver, use_driver=True)
                article["full_text"] = full_text

            text_for_ai = article["title"]
            if snippet:
                text_for_ai += f". {snippet}"
            if full_text:
                text_for_ai += f". {full_text[:1500]}"

            article["analysis"] = groq_analyze(text_for_ai, api_key, model)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    aggregate = aggregate_results(deduped)
    result = {
        "symbol": args.symbol,
        "query": query,
        "avg_score": aggregate["avg_score"],
        "confidence": aggregate["confidence"],
        "sentiment": aggregate["sentiment"],
        "news_count": aggregate["news_count"],
        "details": deduped,
        "generated_at": datetime.now().isoformat(),
        "model": model,
    }

    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News + Sentiment agent with Selenium + BS4 + Groq")
    parser.add_argument("--symbol", help="Symbol, e.g., BBCA.JK", default=None)
    parser.add_argument("--query", help="Search query override", default=None)
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--detail-limit", type=int, default=3)
    parser.add_argument("--out", help="Output JSON file path", default=None)
    parser.add_argument(
        "--headless",
        default="true",
        choices=["true", "false"],
        help="Run Chrome headless (true/false)",
    )

    args = parser.parse_args(argv)
    args.headless = args.headless.lower() == "true"
    if not args.symbol and not args.query:
        parser.error("Either --symbol or --query is required")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    result = run_agent(args)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        _log(f"Wrote output to {out_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
