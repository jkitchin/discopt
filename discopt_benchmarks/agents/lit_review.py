"""
JaxMINLP Weekly Literature Review Agent

An automated agent that monitors the optimization research literature
for new algorithms, techniques, and results relevant to JaxMINLP
development. Runs weekly (via cron or CI) and produces:

  1. A ranked digest of new papers with relevance scores
  2. Actionable recommendations mapped to JaxMINLP components
  3. Implementation proposals for high-impact techniques
  4. Competitive intelligence on other solver developments

Sources monitored:
  - arXiv (math.OC, cs.MS, cs.LG+optimization, cs.DS)
  - Optimization Online
  - Mathematical Programming / Math Programming Computation
  - INFORMS Journal on Computing
  - Operations Research
  - Computers & Chemical Engineering
  - SIAM Journal on Optimization
  - Journal of Global Optimization

Architecture:
  - Search layer: arXiv API, CrossRef, Semantic Scholar, PubMed
  - Analysis layer: LLM (Claude) for relevance scoring and summarization
  - Output layer: Markdown report + structured JSON for tracking

Usage:
    # Run weekly review
    python -m agents.lit_review --weeks 1 --output reports/

    # Review specific topic
    python -m agents.lit_review --query "McCormick relaxation" --weeks 4

    # Full quarterly review
    python -m agents.lit_review --weeks 13 --output reports/ --comprehensive
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# arXiv categories to monitor
ARXIV_CATEGORIES = [
    "math.OC",      # Optimization and Control
    "cs.MS",        # Mathematical Software
    "cs.NA",        # Numerical Analysis (cross-listed as math.NA)
    "cs.LG",        # Machine Learning (for ML+optimization papers)
    "cs.DS",        # Data Structures and Algorithms
    "stat.ML",      # Statistical ML (for Bayesian optimization, UQ)
]

# Keywords that indicate relevance to MINLP solver development
# Organized by JaxMINLP component for targeted routing
RELEVANCE_KEYWORDS = {
    "relaxation": [
        "mccormick", "convex relaxation", "convex envelope",
        "underestimator", "overestimator", "factorable",
        "alphabb", "alpha-bb", "piecewise linear relaxation",
        "polyhedral relaxation", "sdp relaxation", "socp relaxation",
        "quadratic convex reformulation", "rlt",
        "reformulation linearization", "edge-concave",
        "multilinear", "signomial", "convexification",
        "tight relaxation", "relaxation quality",
    ],
    "branching": [
        "branch and bound", "branch-and-bound", "branching strategy",
        "variable selection", "node selection", "strong branching",
        "reliability branching", "learned branching",
        "gnn branching", "graph neural network branch",
        "spatial branching", "disjunctive branching",
        "multi-variable branching",
    ],
    "bound_tightening": [
        "bound tightening", "obbt", "fbbt", "domain reduction",
        "range reduction", "constraint propagation",
        "interval arithmetic", "affine arithmetic",
        "feasibility based bound", "optimization based bound",
        "probing", "preprocessing", "presolve",
    ],
    "cutting_planes": [
        "cutting plane", "valid inequality", "lift and project",
        "disjunctive cut", "intersection cut", "split cut",
        "outer approximation", "extended formulation",
        "perspective cut", "conic cut",
    ],
    "nlp_solver": [
        "interior point method", "ipm", "barrier method",
        "sequential quadratic", "sqp", "trust region",
        "filter method", "augmented lagrangian",
        "nonlinear programming", "ipopt",
    ],
    "lp_solver": [
        "simplex method", "revised simplex", "dual simplex",
        "lu factorization", "basis update", "pricing",
        "sparse linear algebra", "cholesky",
        "linear programming", "highs",
    ],
    "global_optimization": [
        "global optimization", "minlp", "mixed-integer nonlinear",
        "spatial branch", "baron", "couenne", "scip",
        "antigone", "deterministic global",
        "nonconvex", "bilinear", "pooling problem",
        "quadratically constrained", "qcqp", "miqcqp",
    ],
    "ml_for_optimization": [
        "learning to optimize", "machine learning optimization",
        "neural network branch", "gnn optimization",
        "predict and optimize", "decision-focused",
        "differentiable optimization", "implicit differentiation",
        "unrolled optimization", "amortized optimization",
        "neural network constraint", "surrogate optimization",
    ],
    "gpu_parallel": [
        "gpu optimization", "parallel branch and bound",
        "gpu solver", "parallel simplex",
        "accelerated optimization", "cuda optimization",
        "batch optimization", "vectorized",
    ],
    "decomposition": [
        "benders decomposition", "lagrangian relaxation",
        "column generation", "dantzig-wolfe",
        "generalized disjunctive", "gdp",
        "stochastic programming", "robust optimization",
        "two-stage", "scenario decomposition",
    ],
    "software": [
        "optimization software", "solver comparison",
        "benchmark", "minlplib", "performance profile",
        "algebraic modeling", "pyomo", "jump", "gams", "ampl",
    ],
}

# All keywords flattened for initial search
ALL_KEYWORDS = []
for kw_list in RELEVANCE_KEYWORDS.values():
    ALL_KEYWORDS.extend(kw_list)


# ─────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────

class RelevanceLevel(Enum):
    CRITICAL = "critical"       # Directly impacts current phase work
    HIGH = "high"               # Should be implemented or evaluated
    MEDIUM = "medium"           # Worth tracking, potential future use
    LOW = "low"                 # Tangentially related
    BACKGROUND = "background"   # General awareness only


@dataclass
class Paper:
    """A paper identified by the literature review."""
    title: str
    authors: list[str]
    abstract: str
    url: str
    source: str                              # "arxiv", "crossref", "semantic_scholar"
    published: Optional[str] = None          # ISO date string
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    categories: list[str] = field(default_factory=list)

    # Populated by analysis
    relevance: RelevanceLevel = RelevanceLevel.BACKGROUND
    relevance_score: float = 0.0             # 0.0 to 1.0
    matched_components: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    summary: Optional[str] = None            # LLM-generated summary
    implementation_notes: Optional[str] = None  # How to integrate into JaxMINLP
    priority_phase: Optional[str] = None     # Which phase this is most relevant to


@dataclass
class ReviewReport:
    """Complete weekly literature review report."""
    generated_at: str
    period_start: str
    period_end: str
    total_papers_scanned: int = 0
    papers: list[Paper] = field(default_factory=list)
    component_summary: dict[str, list[str]] = field(default_factory=dict)
    action_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for p in d["papers"]:
            p["relevance"] = p["relevance"].value if isinstance(p["relevance"], RelevanceLevel) else p["relevance"]
        return d


# ─────────────────────────────────────────────────────────────
# Search Layer
# ─────────────────────────────────────────────────────────────

class ArxivSearch:
    """Search arXiv for recent papers matching MINLP-related queries."""

    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_SECONDS = 3.0  # arXiv asks for 3s between requests

    def __init__(self):
        self._last_request_time = 0.0

    def search(
        self,
        query: str,
        categories: list[str] = None,
        max_results: int = 100,
        days_back: int = 7,
    ) -> list[Paper]:
        """
        Search arXiv for papers matching query in given categories.

        Args:
            query: Search terms (boolean OR of keywords)
            categories: arXiv categories to restrict to
            max_results: Maximum papers to return
            days_back: Only return papers from last N days

        Returns:
            List of Paper objects
        """
        if categories is None:
            categories = ARXIV_CATEGORIES

        # Build query string
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        search_query = f"({query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "JaxMINLP-LitReview/0.1"})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            self._last_request_time = time.time()
        except Exception as e:
            print(f"  Warning: arXiv request failed: {e}")
            return []

        return self._parse_atom_feed(xml_data, days_back)

    def _parse_atom_feed(self, xml_data: bytes, days_back: int) -> list[Paper]:
        """Parse arXiv Atom XML feed into Paper objects."""
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        root = ET.fromstring(xml_data)
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        papers = []

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            arxiv_id_el = entry.find("atom:id", ns)

            if title is None or summary is None:
                continue

            title_text = " ".join(title.text.strip().split())
            abstract_text = " ".join(summary.text.strip().split())

            # Parse date
            pub_date = None
            if published is not None and published.text:
                try:
                    pub_date = datetime.fromisoformat(
                        published.text.replace("Z", "+00:00")
                    )
                    if pub_date.replace(tzinfo=None) < cutoff:
                        continue  # Too old
                except ValueError:
                    pass

            # Authors
            authors = []
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # arXiv ID and URL
            arxiv_url = ""
            arxiv_id = None
            if arxiv_id_el is not None and arxiv_id_el.text:
                arxiv_url = arxiv_id_el.text.strip()
                # Extract ID from URL like http://arxiv.org/abs/2301.12345v1
                match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", arxiv_url)
                if match:
                    arxiv_id = match.group(1)

            # Categories
            categories = []
            for cat_el in entry.findall("arxiv:primary_category", ns):
                term = cat_el.get("term")
                if term:
                    categories.append(term)
            for cat_el in entry.findall("atom:category", ns):
                term = cat_el.get("term")
                if term and term not in categories:
                    categories.append(term)

            # DOI
            doi = None
            doi_el = entry.find("arxiv:doi", ns)
            if doi_el is not None and doi_el.text:
                doi = doi_el.text.strip()

            papers.append(Paper(
                title=title_text,
                authors=authors,
                abstract=abstract_text,
                url=arxiv_url or f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                source="arxiv",
                published=pub_date.isoformat() if pub_date else None,
                arxiv_id=arxiv_id,
                doi=doi,
                categories=categories,
            ))

        return papers

    def search_by_topics(self, days_back: int = 7, max_per_query: int = 50) -> list[Paper]:
        """
        Run multiple targeted searches covering all JaxMINLP-relevant topics.

        This is the main entry point for the weekly review.
        """
        all_papers: dict[str, Paper] = {}  # Deduplicate by title

        # Core MINLP queries
        queries = [
            # Direct MINLP
            '"mixed-integer nonlinear" OR "spatial branch" OR "global optimization" OR MINLP',
            # Relaxation techniques
            '"convex relaxation" OR "McCormick" OR "convex envelope" OR "underestimator"',
            # Branch and bound
            '"branch and bound" OR "branching strategy" OR "node selection" OR "learned branching"',
            # Bound tightening
            '"bound tightening" OR OBBT OR FBBT OR "domain reduction" OR "constraint propagation"',
            # Cutting planes for nonlinear
            '"cutting plane" OR "valid inequality" OR "outer approximation" AND nonlinear',
            # NLP solvers
            '"interior point" OR "sequential quadratic" OR IPM AND nonlinear',
            # LP / sparse LA
            '"revised simplex" OR "sparse factorization" OR "Cholesky" AND optimization',
            # ML for optimization
            '"learning to optimize" OR "neural network" AND "branch" OR "optimization"',
            # GPU / parallel optimization
            'GPU AND ("branch and bound" OR "optimization" OR "solver")',
            # Software / benchmarks
            'BARON OR SCIP OR Couenne OR "solver comparison" OR "MINLPLib"',
            # Differentiable optimization
            '"differentiable optimization" OR "implicit differentiation" OR "bilevel"',
            # Decomposition
            '"Benders decomposition" OR "Lagrangian" OR "generalized disjunctive"',
        ]

        for i, query in enumerate(queries):
            print(f"  Searching arXiv ({i+1}/{len(queries)}): {query[:60]}...")
            papers = self.search(
                query=query,
                max_results=max_per_query,
                days_back=days_back,
            )
            for p in papers:
                key = p.title.lower().strip()
                if key not in all_papers:
                    all_papers[key] = p
            print(f"    Found {len(papers)} papers ({len(all_papers)} unique total)")

        return list(all_papers.values())


# ─────────────────────────────────────────────────────────────
# Analysis Layer
# ─────────────────────────────────────────────────────────────

class RelevanceAnalyzer:
    """
    Score and categorize papers by relevance to JaxMINLP development.

    Two modes:
      1. Keyword-based scoring (fast, no API calls, always available)
      2. LLM-based scoring (deeper analysis, requires Claude API)
    """

    def __init__(self, use_llm: bool = False, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    def analyze(self, papers: list[Paper]) -> list[Paper]:
        """Score all papers and sort by relevance."""
        for paper in papers:
            self._keyword_score(paper)

        if self.use_llm and self.llm_client:
            # Only send high-keyword-score papers to LLM (cost control)
            candidates = [p for p in papers if p.relevance_score >= 0.3]
            if candidates:
                self._llm_analyze_batch(candidates)

        # Sort by relevance score descending
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    def _keyword_score(self, paper: Paper):
        """
        Fast keyword-based relevance scoring.

        Matches title and abstract against component-specific keyword lists.
        Score is based on: number of components matched, keyword density,
        and title vs abstract weighting (title matches count 3x).
        """
        text_lower = (paper.title + " " + paper.abstract).lower()
        title_lower = paper.title.lower()

        total_score = 0.0
        components_matched = set()
        keywords_matched = set()

        for component, keywords in RELEVANCE_KEYWORDS.items():
            component_score = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in title_lower:
                    component_score += 3.0  # Title match: high signal
                    keywords_matched.add(kw)
                elif kw_lower in text_lower:
                    component_score += 1.0  # Abstract match
                    keywords_matched.add(kw)

            if component_score > 0:
                components_matched.add(component)
                total_score += component_score

        # Normalize to 0-1 range (empirical: 20+ is very high)
        paper.relevance_score = min(1.0, total_score / 20.0)
        paper.matched_components = sorted(components_matched)
        paper.matched_keywords = sorted(keywords_matched)

        # Assign relevance level
        if paper.relevance_score >= 0.7:
            paper.relevance = RelevanceLevel.CRITICAL
        elif paper.relevance_score >= 0.5:
            paper.relevance = RelevanceLevel.HIGH
        elif paper.relevance_score >= 0.3:
            paper.relevance = RelevanceLevel.MEDIUM
        elif paper.relevance_score >= 0.1:
            paper.relevance = RelevanceLevel.LOW
        else:
            paper.relevance = RelevanceLevel.BACKGROUND

    def _llm_analyze_batch(self, papers: list[Paper]):
        """
        Use LLM to produce deeper analysis of high-relevance papers.

        Generates: summary, implementation notes, priority phase mapping.
        """
        for paper in papers:
            prompt = self._build_analysis_prompt(paper)
            try:
                response = self.llm_client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._parse_llm_response(paper, response.content[0].text)
            except Exception as e:
                print(f"  Warning: LLM analysis failed for '{paper.title[:50]}': {e}")

    def _build_analysis_prompt(self, paper: Paper) -> str:
        return f"""You are a research advisor for JaxMINLP, a Rust+JAX hybrid MINLP solver under development.

Analyze this paper for its relevance to JaxMINLP development:

Title: {paper.title}
Authors: {', '.join(paper.authors[:5])}
Abstract: {paper.abstract}

JaxMINLP components this paper may relate to: {', '.join(paper.matched_components)}

Respond in exactly this format (keep each section to 1-3 sentences):

SUMMARY: [What the paper contributes, in plain language]
RELEVANCE: [Why this matters for JaxMINLP specifically]
IMPLEMENTATION: [Concrete next steps if we wanted to use this — which component, estimated effort, dependencies]
PHASE: [Which JaxMINLP phase (1-4) this is most relevant to, or "post-v1.0"]
RISK: [What could go wrong if we implement this, or "none" if straightforward]"""

    def _parse_llm_response(self, paper: Paper, response: str):
        """Parse structured LLM response into paper fields."""
        sections = {}
        current_key = None
        current_lines = []

        for line in response.strip().split("\n"):
            for key in ["SUMMARY", "RELEVANCE", "IMPLEMENTATION", "PHASE", "RISK"]:
                if line.upper().startswith(f"{key}:"):
                    if current_key:
                        sections[current_key] = " ".join(current_lines).strip()
                    current_key = key
                    current_lines = [line[len(key)+1:].strip()]
                    break
            else:
                if current_key:
                    current_lines.append(line.strip())

        if current_key:
            sections[current_key] = " ".join(current_lines).strip()

        paper.summary = sections.get("SUMMARY")
        paper.implementation_notes = sections.get("IMPLEMENTATION")
        paper.priority_phase = sections.get("PHASE")


# ─────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────

class ReportGenerator:
    """Generate markdown and JSON reports from analyzed papers."""

    def generate(
        self,
        papers: list[Paper],
        period_start: str,
        period_end: str,
        total_scanned: int,
        output_dir: Path,
    ) -> ReviewReport:
        """Generate full report and write to files."""

        # Filter to relevant papers only
        relevant = [p for p in papers if p.relevance_score >= 0.1]

        # Build component summary
        component_summary: dict[str, list[str]] = {}
        for p in relevant:
            for comp in p.matched_components:
                if comp not in component_summary:
                    component_summary[comp] = []
                component_summary[comp].append(p.title)

        # Generate action items from critical/high papers
        action_items = []
        for p in relevant:
            if p.relevance in (RelevanceLevel.CRITICAL, RelevanceLevel.HIGH):
                action = f"[{p.relevance.value.upper()}] Review: \"{p.title}\""
                if p.matched_components:
                    action += f" (components: {', '.join(p.matched_components)})"
                if p.implementation_notes:
                    action += f" — {p.implementation_notes}"
                action_items.append(action)

        report = ReviewReport(
            generated_at=datetime.utcnow().isoformat(),
            period_start=period_start,
            period_end=period_end,
            total_papers_scanned=total_scanned,
            papers=relevant,
            component_summary=component_summary,
            action_items=action_items,
        )

        # Write outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        self._write_markdown(report, output_dir / f"lit_review_{date_str}.md")
        self._write_json(report, output_dir / f"lit_review_{date_str}.json")

        return report

    def _write_markdown(self, report: ReviewReport, path: Path):
        """Write human-readable markdown report."""
        lines = []
        lines.append(f"# JaxMINLP Literature Review")
        lines.append(f"")
        lines.append(f"**Period:** {report.period_start} to {report.period_end}")
        lines.append(f"**Generated:** {report.generated_at}")
        lines.append(f"**Papers scanned:** {report.total_papers_scanned}")
        lines.append(f"**Relevant papers:** {len(report.papers)}")
        lines.append(f"")

        # Executive summary
        critical = [p for p in report.papers if p.relevance == RelevanceLevel.CRITICAL]
        high = [p for p in report.papers if p.relevance == RelevanceLevel.HIGH]
        medium = [p for p in report.papers if p.relevance == RelevanceLevel.MEDIUM]

        lines.append(f"## Executive Summary")
        lines.append(f"")
        lines.append(f"- **{len(critical)} critical** papers requiring immediate team review")
        lines.append(f"- **{len(high)} high-relevance** papers with potential implementation value")
        lines.append(f"- **{len(medium)} medium-relevance** papers for background awareness")
        lines.append(f"")

        # Action items
        if report.action_items:
            lines.append(f"## Action Items")
            lines.append(f"")
            for item in report.action_items:
                lines.append(f"- {item}")
            lines.append(f"")

        # Component impact summary
        if report.component_summary:
            lines.append(f"## Papers by JaxMINLP Component")
            lines.append(f"")
            # Sort by number of papers descending
            sorted_components = sorted(
                report.component_summary.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            for component, titles in sorted_components:
                lines.append(f"### {component.replace('_', ' ').title()} ({len(titles)} papers)")
                lines.append(f"")
                for title in titles[:5]:  # Top 5 per component
                    lines.append(f"- {title}")
                if len(titles) > 5:
                    lines.append(f"- ... and {len(titles) - 5} more")
                lines.append(f"")

        # Detailed paper listings by relevance
        for level, label in [
            (RelevanceLevel.CRITICAL, "Critical — Immediate Review Required"),
            (RelevanceLevel.HIGH, "High Relevance"),
            (RelevanceLevel.MEDIUM, "Medium Relevance"),
        ]:
            level_papers = [p for p in report.papers if p.relevance == level]
            if not level_papers:
                continue

            lines.append(f"## {label}")
            lines.append(f"")

            for p in level_papers:
                lines.append(f"### {p.title}")
                lines.append(f"")
                lines.append(f"- **Authors:** {', '.join(p.authors[:5])}")
                if p.published:
                    lines.append(f"- **Published:** {p.published[:10]}")
                lines.append(f"- **Source:** [{p.source}]({p.url})")
                lines.append(f"- **Relevance score:** {p.relevance_score:.2f}")
                lines.append(f"- **Components:** {', '.join(p.matched_components)}")
                lines.append(f"- **Keywords matched:** {', '.join(p.matched_keywords[:8])}")
                lines.append(f"")

                if p.summary:
                    lines.append(f"**Summary:** {p.summary}")
                    lines.append(f"")
                else:
                    # Truncated abstract as fallback
                    abstract_short = p.abstract[:300]
                    if len(p.abstract) > 300:
                        abstract_short += "..."
                    lines.append(f"**Abstract (excerpt):** {abstract_short}")
                    lines.append(f"")

                if p.implementation_notes:
                    lines.append(f"**Implementation notes:** {p.implementation_notes}")
                    lines.append(f"")

                if p.priority_phase:
                    lines.append(f"**Relevant phase:** {p.priority_phase}")
                    lines.append(f"")

                lines.append(f"---")
                lines.append(f"")

        # Footer
        lines.append(f"---")
        lines.append(f"*Generated by JaxMINLP Literature Review Agent. "
                      f"Keyword-based scoring with optional LLM analysis. "
                      f"Review critical papers within 1 week; high-relevance within 2 weeks.*")

        path.write_text("\n".join(lines))
        print(f"  Wrote markdown report: {path}")

    def _write_json(self, report: ReviewReport, path: Path):
        """Write machine-readable JSON report."""

        def serialize(obj):
            if isinstance(obj, RelevanceLevel):
                return obj.value
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Cannot serialize {type(obj)}")

        data = {
            "generated_at": report.generated_at,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "total_papers_scanned": report.total_papers_scanned,
            "summary": {
                "critical": len([p for p in report.papers if p.relevance == RelevanceLevel.CRITICAL]),
                "high": len([p for p in report.papers if p.relevance == RelevanceLevel.HIGH]),
                "medium": len([p for p in report.papers if p.relevance == RelevanceLevel.MEDIUM]),
                "low": len([p for p in report.papers if p.relevance == RelevanceLevel.LOW]),
            },
            "action_items": report.action_items,
            "component_summary": report.component_summary,
            "papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "url": p.url,
                    "arxiv_id": p.arxiv_id,
                    "doi": p.doi,
                    "published": p.published,
                    "relevance": p.relevance.value,
                    "relevance_score": p.relevance_score,
                    "matched_components": p.matched_components,
                    "matched_keywords": p.matched_keywords,
                    "summary": p.summary,
                    "implementation_notes": p.implementation_notes,
                    "priority_phase": p.priority_phase,
                }
                for p in report.papers
            ],
        }

        path.write_text(json.dumps(data, indent=2, default=serialize))
        print(f"  Wrote JSON report: {path}")


# ─────────────────────────────────────────────────────────────
# Main Agent Orchestrator
# ─────────────────────────────────────────────────────────────

class LiteratureReviewAgent:
    """
    Weekly literature review agent for JaxMINLP.

    Orchestrates: search → analyze → report pipeline.

    Usage:
        agent = LiteratureReviewAgent()
        report = agent.run(weeks=1, output_dir=Path("reports/"))

        # With LLM analysis (requires ANTHROPIC_API_KEY):
        agent = LiteratureReviewAgent(use_llm=True)
        report = agent.run(weeks=1)
    """

    def __init__(self, use_llm: bool = False):
        self.arxiv = ArxivSearch()
        self.use_llm = use_llm

        # Initialize LLM client if requested
        self.llm_client = None
        if use_llm:
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic()
                print("  LLM analysis enabled (Claude)")
            except ImportError:
                print("  Warning: anthropic package not installed. "
                      "Using keyword-only scoring.")
                self.use_llm = False

        self.analyzer = RelevanceAnalyzer(
            use_llm=self.use_llm,
            llm_client=self.llm_client,
        )
        self.reporter = ReportGenerator()

    def run(
        self,
        weeks: int = 1,
        output_dir: Path = Path("reports"),
        query: Optional[str] = None,
        comprehensive: bool = False,
    ) -> ReviewReport:
        """
        Run the full literature review pipeline.

        Args:
            weeks: Number of weeks to look back
            output_dir: Directory for output reports
            query: Optional specific query (overrides default topic search)
            comprehensive: If True, increase search breadth and LLM analysis depth

        Returns:
            ReviewReport with all findings
        """
        days_back = weeks * 7
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=days_back)

        print(f"\n{'='*60}")
        print(f"JaxMINLP Literature Review Agent")
        print(f"Period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
        print(f"Mode: {'comprehensive' if comprehensive else 'standard'}, "
              f"LLM: {'enabled' if self.use_llm else 'disabled'}")
        print(f"{'='*60}\n")

        # Step 1: Search
        print("Step 1: Searching arXiv...")
        if query:
            papers = self.arxiv.search(
                query=query,
                days_back=days_back,
                max_results=200 if comprehensive else 100,
            )
        else:
            papers = self.arxiv.search_by_topics(
                days_back=days_back,
                max_per_query=100 if comprehensive else 50,
            )
        total_scanned = len(papers)
        print(f"  Total papers found: {total_scanned}\n")

        if not papers:
            print("  No papers found. Try increasing --weeks or broadening search.")
            return ReviewReport(
                generated_at=datetime.utcnow().isoformat(),
                period_start=period_start.isoformat(),
                period_end=period_end.isoformat(),
            )

        # Step 2: Analyze
        print("Step 2: Analyzing relevance...")
        papers = self.analyzer.analyze(papers)
        relevant = [p for p in papers if p.relevance_score >= 0.1]
        critical = [p for p in papers if p.relevance == RelevanceLevel.CRITICAL]
        high = [p for p in papers if p.relevance == RelevanceLevel.HIGH]
        print(f"  Relevant: {len(relevant)} "
              f"(critical: {len(critical)}, high: {len(high)})\n")

        # Step 3: Report
        print("Step 3: Generating reports...")
        report = self.reporter.generate(
            papers=papers,
            period_start=period_start.strftime("%Y-%m-%d"),
            period_end=period_end.strftime("%Y-%m-%d"),
            total_scanned=total_scanned,
            output_dir=output_dir,
        )

        # Print summary to console
        print(f"\n{'='*60}")
        print(f"Review Complete")
        print(f"{'='*60}")
        print(f"Papers scanned: {total_scanned}")
        print(f"Relevant:       {len(relevant)}")
        print(f"  Critical:     {len(critical)}")
        print(f"  High:         {len(high)}")
        print(f"Action items:   {len(report.action_items)}")

        if critical:
            print(f"\n⚠️  CRITICAL papers requiring immediate review:")
            for p in critical[:5]:
                print(f"  • {p.title}")
                print(f"    Components: {', '.join(p.matched_components)}")
                print(f"    {p.url}")

        if report.action_items:
            print(f"\n📋 Top action items:")
            for item in report.action_items[:5]:
                print(f"  • {item[:120]}")

        print(f"\nFull reports written to: {output_dir}/")
        return report


# ─────────────────────────────────────────────────────────────
# CI/CD Integration
# ─────────────────────────────────────────────────────────────

GITHUB_ACTIONS_WORKFLOW = """
# .github/workflows/lit-review.yml
# Weekly literature review for JaxMINLP
name: Literature Review

on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9:00 UTC
  workflow_dispatch:       # Manual trigger

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install anthropic

      - name: Run literature review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python -m agents.lit_review --weeks 1 --output reports/weekly/

      - name: Upload report artifacts
        uses: actions/upload-artifact@v4
        with:
          name: lit-review-${{ github.run_number }}
          path: reports/weekly/

      - name: Create issue for critical papers
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const glob = require('glob');
            const files = glob.sync('reports/weekly/lit_review_*.json');
            if (files.length === 0) return;

            const report = JSON.parse(fs.readFileSync(files[0], 'utf8'));
            if (report.summary.critical === 0 && report.summary.high === 0) return;

            const body = [
              `## Weekly Literature Review - ${report.period_start} to ${report.period_end}`,
              ``,
              `**Critical papers:** ${report.summary.critical}`,
              `**High-relevance papers:** ${report.summary.high}`,
              ``,
              `### Action Items`,
              ...report.action_items.slice(0, 10).map(a => `- ${a}`),
              ``,
              `See full report in workflow artifacts.`,
            ].join('\\n');

            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `📚 Lit Review: ${report.summary.critical} critical, ${report.summary.high} high-relevance papers`,
              body: body,
              labels: ['literature-review', 'research'],
            });
"""


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="JaxMINLP Weekly Literature Review Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agents.lit_review                          # Default: last 7 days
  python -m agents.lit_review --weeks 4                # Last month
  python -m agents.lit_review --query "pooling problem" --weeks 52  # Targeted search
  python -m agents.lit_review --llm --weeks 1          # With LLM analysis
  python -m agents.lit_review --comprehensive --weeks 13  # Quarterly review
        """,
    )
    parser.add_argument(
        "--weeks", type=int, default=1,
        help="Number of weeks to look back (default: 1)",
    )
    parser.add_argument(
        "--output", type=str, default="reports",
        help="Output directory for reports (default: reports/)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Specific search query (overrides default topic search)",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM-based analysis (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--comprehensive", action="store_true",
        help="Comprehensive mode: broader search, deeper analysis",
    )

    args = parser.parse_args()

    agent = LiteratureReviewAgent(use_llm=args.llm)
    agent.run(
        weeks=args.weeks,
        output_dir=Path(args.output),
        query=args.query,
        comprehensive=args.comprehensive,
    )


if __name__ == "__main__":
    main()
