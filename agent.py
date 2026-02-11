#!/usr/bin/env python3
"""Autonomous multi-agent system with interpreter, orchestrator, and critique loop.

This system processes natural language queries through a pipeline of specialized
agents. The interpreter fixes typos and understands intent. The orchestrator
coordinates research, synthesis, and critique. The critic validates answers
before delivery.

Usage:
    python3 agent.py [options]
    ./agent.sh  # uses environment variables

Architecture:
    User Prompt -> Interpreter -> Orchestrator -> Research/Synthesis -> Critic -> Answer
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# -- System Prompts --

INTERPRETER_PROMPT = """You are a query interpreter. Your job is to:
1. Fix any spelling or grammar errors in the user's input
2. Understand what the user means
3. Reformulate their request into clear, searchable queries

Given the user's message and conversation history, output a JSON object with:
{
  "corrected_prompt": "the user's message with spelling/grammar fixed",
  "intent_type": "question" | "task" | "learning" | "followup",
  "refined_queries": ["specific query 1", "specific query 2"],
  "context_needed": "description of what information is needed",
  "requires_code": true/false,
  "requires_web": true/false,
  "requires_rag": true/false
}

Spelling correction examples:
- "waht is teh lates python" -> "What is the latest Python?"
- "halp me fixx my coed" -> "Help me fix my code"
- "hwo do I uise react" -> "How do I use React?"
- "who wno" -> "Who won?"
- "lern fastapi" -> "Learn FastAPI"

Handle vague inputs:
- "fix my code" -> Look at conversation for code, generate queries about the language/framework
- "that thing" -> Use conversation history to resolve references
- "latest X" -> Convert to "X version 2026" or "X recent updates"
- Pronouns ("it", "this", "that") -> Resolve from context

Be specific. Convert vague questions into precise, searchable queries.
Output ONLY valid JSON, no markdown fences or extra text."""

ORCHESTRATOR_PROMPT = """You are an orchestrator agent coordinating a multi-agent system.

Current intent:
- Type: {intent_type}
- Queries: {queries}
- Context needed: {context}
- Requires web: {requires_web}
- Requires RAG: {requires_rag}

You have these tools available:
- search_rag: Query indexed documents in the knowledge base
- search_web: Search the web for current information
- ingest_url: Index a URL into the knowledge base
- ingest_crawl: Crawl and index a documentation site (max 50 pages)
- delegate_research: Have the research worker summarize findings
- delegate_synthesis: Have the synthesis worker generate an answer
- answer: Provide the final response to the user

Plan your approach:
1. For questions requiring current info, use search_web first
2. For questions about indexed docs, use search_rag
3. For learning intents, consider ingest_crawl to index docs first
4. Always delegate_synthesis before providing final answer
5. The critic will review synthesis output automatically

Call tools by responding with JSON:
{"tool": "tool_name", "args": {"arg1": "value1"}}

When ready to give final answer:
{"tool": "answer", "args": {"response": "your final answer"}}"""

RESEARCH_PROMPT = """You are a research worker. Summarize the provided search results into clear, factual bullet points.

Search results:
{results}

Original queries: {queries}

Provide a concise summary with:
- Key facts found
- Source URLs for citations
- Any gaps in the information

Format as bullet points. Be factual, cite sources."""

SYNTHESIS_PROMPT = """You are a synthesis worker. Generate a clear, helpful answer based on the research provided.

Research summary:
{research}

Original question: {question}
Context: {context}

Generate a complete answer that:
- Directly addresses the question
- Uses information from the research
- Is clear and actionable
- Includes code examples if relevant
- Cites sources when making claims

Be concise but thorough."""

CRITIC_PROMPT = """You are a quality assurance agent. Review the draft answer and determine if it should be approved or revised.

Original question: {question}
Research summary: {research}
Draft answer: {draft}

Check for:
1. Does the answer address the question?
2. Are claims supported by the research summary?
3. Is anything important missing?
4. Are there any obvious errors or contradictions?
5. Is the answer clear and actionable?

Output JSON only:
{
  "approved": true if ready to show user, false if needs revision,
  "issues": ["list problems found"],
  "suggestions": ["specific improvements"],
  "confidence": 0.0-1.0 how confident you are in this assessment
}

If the answer is good enough (80%+ of requirements met), approve it. Perfect is the enemy of good.
Output ONLY valid JSON, no markdown fences or extra text."""


@dataclass
class AgentConfig:
    """Configuration for the multi-agent system."""
    ollama_url: str = "http://localhost:11434"
    searxng_url: str = "http://localhost:8080"
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "nomic-embed-text"
    interpreter_model: str = "qwen2.5:7b"
    orchestrator_model: str = "qwen3:14b-agent"
    research_model: str = "qwen2.5:7b"
    synthesis_model: str = "qwen3:14b-agent"
    critic_model: str = "qwen2.5:7b"
    max_iterations: int = 10
    max_revisions: int = 3
    collection: str = "default"


@dataclass
class Intent:
    """Structured intent from the interpreter."""
    corrected_prompt: str
    intent_type: str
    refined_queries: list
    context_needed: str
    requires_code: bool = False
    requires_web: bool = True
    requires_rag: bool = False


@dataclass
class CritiqueResult:
    """Result from the critic agent."""
    approved: bool
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    confidence: float = 0.0


class OllamaClient:
    """Simple client for Ollama API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(self, model: str, messages: list, stream: bool = False) -> str:
        """Send chat request and return response content."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 404:
            return self._chat_openai_compat(model, messages, stream)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def embed(self, model: str, text: str) -> list:
        """Generate embeddings for text."""
        url = f"{self.base_url}/api/embed"
        payload = {"model": model, "input": text}
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 404:
            return self._embed_openai_compat(model, text)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [[]])
        return embeddings[0] if embeddings else []

    def _chat_openai_compat(self, model: str, messages: list, stream: bool = False) -> str:
        """Fallback for OpenAI-compatible chat endpoints."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _embed_openai_compat(self, model: str, text: str) -> list:
        """Fallback for OpenAI-compatible embedding endpoints."""
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": model, "input": text}
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return []
        return data[0].get("embedding", [])


class QdrantClient:
    """Simple client for Qdrant vector database."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def ensure_collection(self, name: str, vector_size: int = 768):
        """Create collection if it doesn't exist."""
        url = f"{self.base_url}/collections/{name}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass

        payload = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        }
        response = requests.put(url, json=payload, timeout=30)
        return response.status_code in (200, 201)

    def upsert(self, collection: str, points: list):
        """Insert or update points."""
        url = f"{self.base_url}/collections/{collection}/points"
        payload = {"points": points}
        response = requests.put(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def search(self, collection: str, vector: list, limit: int = 5) -> list:
        """Search for similar vectors."""
        url = f"{self.base_url}/collections/{collection}/points/search"
        payload = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("result", [])


class InterpreterAgent:
    """Interprets user intent and fixes typos."""

    def __init__(self, ollama: OllamaClient, model: str):
        self.ollama = ollama
        self.model = model

    def interpret(self, user_prompt: str, conversation: list) -> Intent:
        """Interpret user prompt and return structured intent."""
        conv_text = self._format_conversation(conversation)

        messages = [
            {"role": "system", "content": INTERPRETER_PROMPT},
            {"role": "user", "content": f"Conversation history:\n{conv_text}\n\nUser message:\n{user_prompt}"}
        ]

        response = self.ollama.chat(self.model, messages)
        parsed = self._parse_json(response)

        return Intent(
            corrected_prompt=parsed.get("corrected_prompt", user_prompt),
            intent_type=parsed.get("intent_type", "question"),
            refined_queries=parsed.get("refined_queries", [user_prompt]),
            context_needed=parsed.get("context_needed", ""),
            requires_code=parsed.get("requires_code", False),
            requires_web=parsed.get("requires_web", True),
            requires_rag=parsed.get("requires_rag", False),
        )

    def _format_conversation(self, conversation: list) -> str:
        """Format conversation history for context."""
        if not conversation:
            return "(no previous conversation)"
        lines = []
        for msg in conversation[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:500]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from model response."""
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}


class IngestWorker:
    """Ingests content into the vector database."""

    def __init__(self, ollama: OllamaClient, qdrant: QdrantClient,
                 embedding_model: str, collection: str = "default"):
        self.ollama = ollama
        self.qdrant = qdrant
        self.embedding_model = embedding_model
        self.collection = collection
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def fetch_url(self, url: str) -> str:
        """Fetch and index a single URL."""
        text = self._scrape(url)
        if not text:
            return f"Failed to fetch {url}"

        chunks = self._chunk(text)
        points = []

        for i, chunk in enumerate(chunks):
            embedding = self.ollama.embed(self.embedding_model, chunk)
            if not embedding:
                continue

            point_id = hash(f"{url}:{i}") & 0x7FFFFFFFFFFFFFFF
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    "url": url,
                    "chunk_index": i,
                }
            })

        if points:
            self.qdrant.ensure_collection(self.collection, len(points[0]["vector"]))
            self.qdrant.upsert(self.collection, points)

        return f"Indexed {len(points)} chunks from {url}"

    def crawl_site(self, start_url: str, max_pages: int = 50) -> str:
        """Crawl a site and index all pages."""
        visited = set()
        to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        total_chunks = 0

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            visited.add(url)
            try:
                text, links = self._scrape_with_links(url)
                if not text:
                    continue

                chunks = self._chunk(text)
                points = []

                for i, chunk in enumerate(chunks):
                    embedding = self.ollama.embed(self.embedding_model, chunk)
                    if not embedding:
                        continue

                    point_id = hash(f"{url}:{i}") & 0x7FFFFFFFFFFFFFFF
                    points.append({
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "text": chunk,
                            "url": url,
                            "chunk_index": i,
                        }
                    })

                if points:
                    self.qdrant.ensure_collection(self.collection, len(points[0]["vector"]))
                    self.qdrant.upsert(self.collection, points)
                    total_chunks += len(points)

                for link in links:
                    if urlparse(link).netloc == base_domain and link not in visited:
                        to_visit.append(link)

            except requests.RequestException:
                continue

        return f"Indexed {total_chunks} chunks from {len(visited)} pages"

    def _scrape(self, url: str) -> str:
        """Fetch URL and extract text."""
        try:
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AIAgent/1.0)"
            })
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            return soup.get_text(separator="\n", strip=True)
        except requests.RequestException:
            return ""

    def _scrape_with_links(self, url: str) -> tuple:
        """Fetch URL and extract text and links."""
        try:
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AIAgent/1.0)"
            })
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            links = []
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                if href.startswith("/"):
                    href = urljoin(url, href)
                if href.startswith("http"):
                    links.append(href.split("#")[0])

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return text, links
        except requests.RequestException:
            return "", []

    def _chunk(self, text: str) -> list:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks


class ResearchWorker:
    """Searches and summarizes information."""

    def __init__(self, ollama: OllamaClient, qdrant: QdrantClient,
                 searxng_url: str, embedding_model: str, model: str,
                 collection: str = "default"):
        self.ollama = ollama
        self.qdrant = qdrant
        self.searxng_url = searxng_url.rstrip("/")
        self.embedding_model = embedding_model
        self.model = model
        self.collection = collection

    def search_and_summarize(self, queries: list,
                             use_rag: bool = False,
                             use_web: bool = True) -> str:
        """Search sources and summarize findings."""
        results = []

        if use_rag:
            for query in queries:
                rag_results = self._search_rag(query)
                results.extend(rag_results)

        if use_web:
            for query in queries:
                web_results = self._search_web(query)
                results.extend(web_results)

        if not results:
            return "No results found."

        results_text = "\n\n".join([
            f"Source: {r.get('url', 'unknown')}\n{r.get('text', '')[:1000]}"
            for r in results[:10]
        ])

        messages = [
            {"role": "system", "content": RESEARCH_PROMPT.format(
                results=results_text,
                queries=", ".join(queries)
            )},
            {"role": "user", "content": "Summarize the search results."}
        ]

        return self.ollama.chat(self.model, messages)

    def _search_rag(self, query: str) -> list:
        """Search Qdrant for relevant documents."""
        try:
            embedding = self.ollama.embed(self.embedding_model, query)
            if not embedding:
                return []

            results = self.qdrant.search(self.collection, embedding, limit=5)
            return [
                {
                    "url": r.get("payload", {}).get("url", "indexed"),
                    "text": r.get("payload", {}).get("text", ""),
                }
                for r in results
            ]
        except requests.RequestException:
            return []

    def _search_web(self, query: str) -> list:
        """Search SearXNG for web results."""
        try:
            url = f"{self.searxng_url}/search"
            params = {"q": query, "format": "json"}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return [
                {
                    "url": r.get("url", ""),
                    "text": f"{r.get('title', '')}\n{r.get('content', '')}",
                }
                for r in data.get("results", [])[:5]
            ]
        except requests.RequestException:
            return []


class SynthesisWorker:
    """Generates answers from research."""

    def __init__(self, ollama: OllamaClient, model: str):
        self.ollama = ollama
        self.model = model

    def generate_answer(self, question: str, research: str, context: str = "") -> str:
        """Generate an answer based on research."""
        messages = [
            {"role": "system", "content": SYNTHESIS_PROMPT.format(
                research=research,
                question=question,
                context=context or "(no additional context)"
            )},
            {"role": "user", "content": f"Answer: {question}"}
        ]

        return self.ollama.chat(self.model, messages)

    def revise(self, draft: str, suggestions: list) -> str:
        """Revise a draft based on suggestions."""
        messages = [
            {"role": "system", "content": "Revise the draft to address these issues."},
            {"role": "user", "content": f"Draft:\n{draft}\n\nIssues to fix:\n" +
                                        "\n".join(f"- {s}" for s in suggestions)}
        ]

        return self.ollama.chat(self.model, messages)


class CriticAgent:
    """Validates answers before delivery."""

    def __init__(self, ollama: OllamaClient, model: str):
        self.ollama = ollama
        self.model = model

    def review(self, draft: str, research: str, question: str) -> CritiqueResult:
        """Review a draft answer."""
        messages = [
            {"role": "system", "content": CRITIC_PROMPT.format(
                question=question,
                research=research,
                draft=draft
            )},
            {"role": "user", "content": "Review the draft answer."}
        ]

        response = self.ollama.chat(self.model, messages)
        parsed = self._parse_json(response)

        return CritiqueResult(
            approved=parsed.get("approved", True),
            issues=parsed.get("issues", []),
            suggestions=parsed.get("suggestions", []),
            confidence=parsed.get("confidence", 0.8),
        )

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from model response."""
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"approved": True, "confidence": 0.5}


class OrchestratorAgent:
    """Orchestrates the multi-agent workflow."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.ollama = OllamaClient(config.ollama_url)
        self.qdrant = QdrantClient(config.qdrant_url)

        self.interpreter = InterpreterAgent(self.ollama, config.interpreter_model)
        self.ingest = IngestWorker(
            self.ollama, self.qdrant,
            config.embedding_model, config.collection
        )
        self.research = ResearchWorker(
            self.ollama, self.qdrant,
            config.searxng_url, config.embedding_model,
            config.research_model, config.collection
        )
        self.synthesis = SynthesisWorker(self.ollama, config.synthesis_model)
        self.critic = CriticAgent(self.ollama, config.critic_model)

    def run(self, user_prompt: str, conversation: list) -> tuple:
        """Run the full agent pipeline.

        Returns tuple of (answer, intent) for conversation tracking.
        """
        # Step 1: Interpret user intent
        intent = self.interpreter.interpret(user_prompt, conversation)
        self._log(f"Corrected: {intent.corrected_prompt}")
        self._log(f"Intent: {intent.intent_type}")

        # Step 2: Handle learning intents with crawling
        if intent.intent_type == "learning" and intent.refined_queries:
            for query in intent.refined_queries:
                if query.startswith("http"):
                    self._log(f"Crawling: {query}...")
                    result = self.ingest.crawl_site(query)
                    self._log(f"  {result}")
                    intent.requires_rag = True

        # Step 3: Research
        self._log("Researching...")
        research_summary = self.research.search_and_summarize(
            intent.refined_queries,
            use_rag=intent.requires_rag,
            use_web=intent.requires_web
        )

        # Step 4: Synthesis
        self._log("Synthesizing...")
        draft = self.synthesis.generate_answer(
            intent.corrected_prompt,
            research_summary,
            intent.context_needed
        )

        # Step 5: Critique loop
        final_answer = self._run_critique_loop(
            draft, research_summary, intent.corrected_prompt
        )

        return final_answer, intent

    def _run_critique_loop(self, draft: str, research: str, question: str) -> str:
        """Run bounded critique and revision loop."""
        for attempt in range(self.config.max_revisions):
            critique = self.critic.review(draft, research, question)

            if critique.approved:
                self._log(f"Critic: approved (confidence: {critique.confidence:.0%})")
                return draft

            if attempt < self.config.max_revisions - 1:
                issue_summary = critique.issues[0] if critique.issues else "needs revision"
                self._log(f"Critic: revision needed - {issue_summary}")
                draft = self.synthesis.revise(draft, critique.suggestions)
            else:
                self._log("Critic: max revisions reached, returning best effort")

        return draft

    def ingest_url(self, url: str) -> str:
        """Index a single URL."""
        self._log(f"Indexing: {url}...")
        return self.ingest.fetch_url(url)

    def ingest_crawl(self, url: str, max_pages: int = 50) -> str:
        """Crawl and index a site."""
        self._log(f"Crawling: {url} (max {max_pages} pages)...")
        return self.ingest.crawl_site(url, max_pages)

    def _log(self, message: str):
        """Print status message."""
        print(f"  [{message}]")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous multi-agent system"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--searxng-url",
        default="http://localhost:8080",
        help="SearXNG search URL",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant vector database URL",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model for RAG",
    )
    parser.add_argument(
        "--interpreter-model",
        default="qwen2.5:7b",
        help="Model for interpreter agent",
    )
    parser.add_argument(
        "--orchestrator-model",
        default="qwen3:14b-agent",
        help="Model for orchestrator agent",
    )
    parser.add_argument(
        "--research-model",
        default="qwen2.5:7b",
        help="Model for research worker",
    )
    parser.add_argument(
        "--synthesis-model",
        default="qwen3:14b-agent",
        help="Model for synthesis worker",
    )
    parser.add_argument(
        "--critic-model",
        default="qwen2.5:7b",
        help="Model for critic agent",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--ingest",
        metavar="URL",
        help="Index a URL and exit",
    )
    parser.add_argument(
        "--crawl",
        metavar="URL",
        help="Crawl a site and exit",
    )
    args = parser.parse_args()

    config = AgentConfig(
        ollama_url=args.ollama_url,
        searxng_url=args.searxng_url,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model,
        interpreter_model=args.interpreter_model,
        orchestrator_model=args.orchestrator_model,
        research_model=args.research_model,
        synthesis_model=args.synthesis_model,
        critic_model=args.critic_model,
        collection=args.collection,
    )

    orchestrator = OrchestratorAgent(config)

    # Handle one-shot ingest commands
    if args.ingest:
        result = orchestrator.ingest_url(args.ingest)
        print(result)
        return

    if args.crawl:
        result = orchestrator.ingest_crawl(args.crawl)
        print(result)
        return

    # Interactive loop
    print()
    print("=" * 60)
    print("  Autonomous Multi-Agent System")
    print("=" * 60)
    print(f"  Interpreter: {config.interpreter_model}")
    print(f"  Synthesis:   {config.synthesis_model}")
    print(f"  Research:    {config.research_model}")
    print(f"  Critic:      {config.critic_model}")
    print()
    print("  Commands:")
    print("    /ingest <url>  - Index a URL")
    print("    /crawl <url>   - Crawl and index a site")
    print("    /quit          - Exit")
    print()

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        # Handle slash commands
        if user_input.startswith("/ingest "):
            url = user_input[8:].strip()
            result = orchestrator.ingest_url(url)
            print(f"\nAgent: {result}\n")
            continue

        if user_input.startswith("/crawl "):
            url = user_input[7:].strip()
            result = orchestrator.ingest_crawl(url)
            print(f"\nAgent: {result}\n")
            continue

        # Run the agent pipeline
        start_time = time.time()
        try:
            answer, intent = orchestrator.run(user_input, conversation)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            failing_url = exc.request.url if exc.request is not None else config.ollama_url
            print(f"\nAgent error: HTTP {status} from {failing_url}")
            if status == 404 and "/api/" in str(failing_url):
                print("Hint: This endpoint may only support /v1 APIs. Set OLLAMA_URL to a native Ollama endpoint or keep this fallback in place.")
            print()
            continue
        except requests.RequestException as exc:
            print(f"\nAgent error: network request failed: {exc}\n")
            continue
        elapsed = time.time() - start_time

        print(f"\nAgent: {answer}")
        print(f"  [{elapsed:.1f}s]\n")

        # Update conversation history
        conversation.append({"role": "user", "content": intent.corrected_prompt})
        conversation.append({"role": "assistant", "content": answer})

        # Keep conversation bounded
        if len(conversation) > 20:
            conversation = conversation[-20:]


if __name__ == "__main__":
    main()
