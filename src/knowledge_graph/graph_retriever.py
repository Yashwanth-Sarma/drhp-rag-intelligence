"""
src/knowledge_graph/graph_retriever.py

Queries Neo4j via HTTP Query API for relationship and comparison questions.
Uses HTTPS (port 443) — same approach as graph_builder.py.
"""

import logging
import requests
from typing import Optional

from src.configuration.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from src.shared.exceptions import FinSightError

logger = logging.getLogger(__name__)


def _build_http_url(neo4j_uri: str, instance_id: str) -> str:
    host = neo4j_uri.replace("neo4j+s://", "").replace("neo4j://", "").rstrip("/")
    return f"https://{host}/db/{instance_id}/query/v2"


class GraphRetriever:
    """
    Queries Neo4j knowledge graph via HTTP Query API.
    Returns structured relationship data for multi-hop queries.
    """

    def __init__(self) -> None:
        if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
            raise FinSightError("NEO4J credentials missing from .env")

        self.instance_id = NEO4J_USERNAME
        self.api_url = _build_http_url(NEO4J_URI, self.instance_id)
        self.auth = (NEO4J_USERNAME, NEO4J_PASSWORD)
        logger.info("GraphRetriever initialized", extra={"url": self.api_url})

    def _run_query(self, cypher: str, parameters: Optional[dict] = None) -> list[dict]:
        """Run Cypher query, return list of result dicts."""
        payload = {"statement": cypher}
        if parameters:
            payload["parameters"] = parameters

        try:
            response = requests.post(
                self.api_url,
                auth=self.auth,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if not response.ok:
                logger.warning(f"Graph query failed {response.status_code}: {response.text[:100]}")
                return []

            result = response.json()
            data = result.get("data", {})
            fields = data.get("fields", [])
            values = data.get("values", [])
            return [dict(zip(fields, row)) for row in values]

        except Exception as e:
            logger.warning(f"Graph query error: {e}")
            return []

    def get_entity_relationships(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
    ) -> list[dict]:
        """Get all relationships for a given entity name."""
        if relation_type:
            cypher = """
            MATCH (a:Entity {name: $name})-[r:RELATED_TO {type: $rel_type}]->(b:Entity)
            RETURN a.name AS from_entity, r.type AS relation,
                   b.name AS to_entity, r.context AS context
            LIMIT 20
            """
            return self._run_query(cypher, {"name": entity_name, "rel_type": relation_type})
        else:
            cypher = """
            MATCH (a:Entity {name: $name})-[r:RELATED_TO]->(b:Entity)
            RETURN a.name AS from_entity, r.type AS relation,
                   b.name AS to_entity, r.context AS context
            LIMIT 30
            """
            return self._run_query(cypher, {"name": entity_name})

    def compare_companies(self, company_a: str, company_b: str) -> dict:
        """Compare relationships for two companies."""
        rels_a = self.get_entity_relationships(company_a)
        rels_b = self.get_entity_relationships(company_b)
        entities_a = {r["to_entity"] for r in rels_a}
        entities_b = {r["to_entity"] for r in rels_b}
        return {
            "company_a": company_a,
            "company_b": company_b,
            "company_a_relations": rels_a,
            "company_b_relations": rels_b,
            "shared_entities": list(entities_a & entities_b),
        }

    def search_entities(self, search_term: str) -> list[dict]:
        """Find entities by partial name match."""
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($term)
        RETURN e.name AS name, e.type AS type, e.source_company AS company
        LIMIT 10
        """
        return self._run_query(cypher, {"term": search_term})

    def format_for_llm(self, relationships: list[dict]) -> str:
        """Format graph results as readable context string for LLM."""
        if not relationships:
            return "No graph relationships found for this query."
        lines = ["Knowledge Graph Evidence:"]
        for r in relationships:
            context = f" ({r['context']})" if r.get("context") else ""
            lines.append(
                f"  {r['from_entity']} --[{r['relation']}]--> {r['to_entity']}{context}"
            )
        return "\n".join(lines)