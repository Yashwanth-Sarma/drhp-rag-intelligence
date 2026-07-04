"""
src/knowledge_graph/graph_retriever.py

Queries Neo4j to answer relationship and comparison questions.
Used by Stage 3 pipeline for multi-hop queries that flat vector search cannot handle.

Example queries this handles:
- "What subsidiaries does Zomato own?"
- "What companies compete with Paytm?"
- "How is Hyperpure related to Zomato?"
"""

from typing import Optional

from src.configuration.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from src.shared.logger import get_logger
from src.shared.exceptions import FinSightError

logger = get_logger(__name__)


class GraphRetriever:
    """
    Queries the Neo4j knowledge graph for entity relationships.
    Returns structured results that feed into the LLM answer generator.
    """

    def __init__(self) -> None:
        if not all([NEO4J_URI, NEO4J_PASSWORD]):
            raise FinSightError("NEO4J credentials missing from .env")
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logger.info("GraphRetriever connected to Neo4j")
        except Exception as e:
            raise FinSightError(f"Neo4j connection failed: {e}") from e

    def close(self) -> None:
        self.driver.close()

    def get_entity_relationships(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
        depth: int = 2,
    ) -> list[dict]:
        """
        Get all relationships for a given entity.

        Args:
            entity_name:   Name of the entity to query (e.g. "Zomato")
            relation_type: Filter by relationship type (e.g. "OWNS")
            depth:         How many hops to traverse (default 2)

        Returns:
            List of relationship dicts: {from, relation, to, context}
        """
        if relation_type:
            query = """
            MATCH (a:Entity {name: $name})-[r:RELATED_TO {type: $rel_type}]->(b:Entity)
            RETURN a.name AS from_entity, r.type AS relation,
                   b.name AS to_entity, r.context AS context
            LIMIT 20
            """
            params = {"name": entity_name, "rel_type": relation_type}
        else:
            query = """
            MATCH (a:Entity {name: $name})-[r:RELATED_TO]->(b:Entity)
            RETURN a.name AS from_entity, r.type AS relation,
                   b.name AS to_entity, r.context AS context
            LIMIT 30
            """
            params = {"name": entity_name}

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def compare_companies(
        self,
        company_a: str,
        company_b: str,
        relation_type: Optional[str] = None,
    ) -> dict:
        """
        Compare relationships for two companies.

        Returns:
            dict with 'company_a_relations', 'company_b_relations', 'shared_entities'
        """
        rels_a = self.get_entity_relationships(company_a, relation_type)
        rels_b = self.get_entity_relationships(company_b, relation_type)

        entities_a = {r["to_entity"] for r in rels_a}
        entities_b = {r["to_entity"] for r in rels_b}
        shared = entities_a & entities_b

        return {
            "company_a": company_a,
            "company_b": company_b,
            "company_a_relations": rels_a,
            "company_b_relations": rels_b,
            "shared_entities": list(shared),
        }

    def search_entities(self, search_term: str) -> list[dict]:
        """Find entities by partial name match."""
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($term)
        RETURN e.name AS name, e.type AS type, e.source_company AS company
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, term=search_term)
            return [dict(record) for record in result]

    def format_for_llm(self, relationships: list[dict]) -> str:
        """Format graph results as readable context for LLM answer generation."""
        if not relationships:
            return "No graph relationships found."

        lines = ["Knowledge Graph Evidence:"]
        for r in relationships:
            lines.append(
                f"  {r['from_entity']} --[{r['relation']}]--> {r['to_entity']}"
                + (f" ({r['context']})" if r.get("context") else "")
            )
        return "\n".join(lines)


if __name__ == "__main__":
    retriever = GraphRetriever()

    print("Searching for Zomato...")
    rels = retriever.get_entity_relationships("Zomato")
    print(f"Found {len(rels)} relationships")
    for r in rels[:5]:
        print(f"  {r['from_entity']} --[{r['relation']}]--> {r['to_entity']}")

    print("\nFormatted for LLM:")
    print(retriever.format_for_llm(rels[:3]))
    retriever.close()