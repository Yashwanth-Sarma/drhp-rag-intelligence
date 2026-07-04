"""
src/knowledge_graph/graph_builder.py

Loads extracted entities from JSONL file into Neo4j AuraDB.

Run this AFTER entity_extractor.py has processed chunks.
It reads data/knowledge_graph/extracted_entities.jsonl
and creates nodes + relationships in Neo4j.

Why Neo4j?
Enables Cypher queries like:
  MATCH (c:Company {name: "Zomato"})-[:OWNS]->(s:Subsidiary)
  RETURN s.name
which flat vector search cannot answer.
"""

import json
from pathlib import Path

from src.configuration.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, DATA_DIR
from src.shared.logger import get_logger
from src.shared.exceptions import FinSightError

logger = get_logger(__name__)

ENTITIES_FILE = DATA_DIR / "knowledge_graph" / "extracted_entities.jsonl"

# Valid entity types and relationship types
VALID_ENTITY_TYPES = {
    "COMPANY", "PERSON", "METRIC", "PRODUCT",
    "REGULATOR", "SUBSIDIARY", "SECTOR"
}
VALID_RELATION_TYPES = {
    "OWNS", "COMPETES_WITH", "INVESTED_IN",
    "REGULATED_BY", "REPORTED", "ACQUIRED",
    "PART_OF", "SERVES", "OPERATES_IN"
}


class GraphBuilder:
    """
    Loads extracted entities into Neo4j.
    Creates nodes for each entity and edges for each relationship.
    Idempotent — safe to run multiple times (uses MERGE not CREATE).
    """

    def __init__(self) -> None:
        if not all([NEO4J_URI, NEO4J_PASSWORD]):
            raise FinSightError(
                "NEO4J_URI and NEO4J_PASSWORD required. "
                "Check your .env file."
            )
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logger.info("GraphBuilder connected to Neo4j")
        except Exception as e:
            raise FinSightError(f"Neo4j connection failed: {e}") from e

    def close(self) -> None:
        self.driver.close()

    def clear_graph(self) -> None:
        """Delete all nodes and relationships. USE WITH CAUTION."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Neo4j graph cleared.")

    def create_indexes(self) -> None:
        """Create Neo4j indexes for faster queries."""
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)")
        logger.info("Neo4j indexes created.")

    def _merge_entity(self, session, entity: dict, source_company: str, doc_type: str, year: str) -> None:
        """Create or update an entity node."""
        name = entity.get("name", "").strip()
        etype = entity.get("type", "COMPANY").upper()

        if not name or etype not in VALID_ENTITY_TYPES:
            return

        session.run(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type,
                e.source_company = $source_company,
                e.doc_type = $doc_type,
                e.year = $year
            """,
            name=name,
            type=etype,
            source_company=source_company,
            doc_type=doc_type,
            year=year,
        )

    def _merge_relationship(self, session, rel: dict, source_company: str) -> None:
        """Create or update a relationship between two entities."""
        from_name = rel.get("from", "").strip()
        to_name = rel.get("to", "").strip()
        relation = rel.get("relation", "").upper().replace(" ", "_")
        context = rel.get("context", "")

        if not from_name or not to_name or relation not in VALID_RELATION_TYPES:
            return

        # Use dynamic relationship type via APOC workaround
        # Standard Cypher doesn't support dynamic relationship types
        # so we use a generic HAS_RELATION with a type property
        session.run(
            """
            MERGE (a:Entity {name: $from_name})
            MERGE (b:Entity {name: $to_name})
            MERGE (a)-[r:RELATED_TO {type: $relation}]->(b)
            SET r.context = $context,
                r.source_company = $source_company
            """,
            from_name=from_name,
            to_name=to_name,
            relation=relation,
            context=context,
            source_company=source_company,
        )

    def load_from_jsonl(self, entities_file: Path = None) -> dict:
        """
        Load all extracted entities from JSONL into Neo4j.
        Uses MERGE so safe to run multiple times — won't create duplicates.

        Returns:
            Summary dict with node and relationship counts.
        """
        file_path = entities_file or ENTITIES_FILE

        if not file_path.exists():
            raise FinSightError(
                f"Entities file not found: {file_path}. "
                "Run entity_extractor.py first."
            )

        self.create_indexes()

        total_entities = 0
        total_relationships = 0
        records_processed = 0

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        print(f"\nLoading {len(lines)} chunk records into Neo4j...")

        with self.driver.session() as session:
            for i, line in enumerate(lines, 1):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                company = record.get("company_name", "Unknown")
                doc_type = record.get("doc_type", "Unknown")
                year = record.get("year", "Unknown")

                for entity in record.get("entities", []):
                    self._merge_entity(session, entity, company, doc_type, year)
                    total_entities += 1

                for rel in record.get("relationships", []):
                    self._merge_relationship(session, rel, company)
                    total_relationships += 1

                records_processed += 1

                if i % 100 == 0:
                    print(f"  [{i}/{len(lines)}] processed...")

        summary = {
            "records_processed": records_processed,
            "total_entities_merged": total_entities,
            "total_relationships_merged": total_relationships,
        }

        logger.info("Graph loading complete", extra=summary)
        print(f"\nNeo4j loading complete:")
        print(f"  Records processed:    {records_processed}")
        print(f"  Entities merged:      {total_entities}")
        print(f"  Relationships merged: {total_relationships}")

        return summary

    def get_stats(self) -> dict:
        """Return count of nodes and relationships in Neo4j."""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        return {"nodes": nodes, "relationships": rels}


if __name__ == "__main__":
    builder = GraphBuilder()
    stats_before = builder.get_stats()
    print(f"Before: {stats_before}")

    summary = builder.load_from_jsonl()

    stats_after = builder.get_stats()
    print(f"After: {stats_after}")
    builder.close()