"""
src/knowledge_graph/graph_builder.py

Loads extracted entities from JSONL into Neo4j AuraDB.
Uses the HTTP Query API (HTTPS port 443) instead of the Bolt protocol (port 7687).

Why HTTP API instead of Bolt driver?
Port 7687 (Bolt) is blocked on many home networks, ISPs, and Windows firewalls.
The Neo4j HTTP Query API uses HTTPS (port 443) which is always open.
Functionally identical for our use case — same Cypher queries, same results.

Run AFTER extract_entities.py has processed chunks.
Input:  data/knowledge_graph/extracted_entities.jsonl
Output: Neo4j graph with entity nodes and relationship edges
"""

import json
import logging
import os
import requests
from pathlib import Path
from typing import Optional

from src.configuration.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    DATA_DIR,
)
from src.shared.exceptions import FinSightError

logger = logging.getLogger(__name__)

ENTITIES_FILE = DATA_DIR / "knowledge_graph" / "extracted_entities.jsonl"

VALID_ENTITY_TYPES = {
    "COMPANY", "PERSON", "METRIC", "PRODUCT",
    "REGULATOR", "SUBSIDIARY", "SECTOR",
}
VALID_RELATION_TYPES = {
    "OWNS", "COMPETES_WITH", "INVESTED_IN",
    "REGULATED_BY", "REPORTED", "ACQUIRED",
    "PART_OF", "SERVES", "OPERATES_IN", "RELATED_TO",
}


def _build_http_url(neo4j_uri: str, instance_id: str) -> str:
    """
    Build the HTTP Query API URL from the neo4j+s:// URI.

    neo4j+s://2f5a3872.databases.neo4j.io
    → https://2f5a3872.databases.neo4j.io/db/2f5a3872/query/v2
    """
    host = neo4j_uri.replace("neo4j+s://", "").replace("neo4j://", "").rstrip("/")
    return f"https://{host}/db/{instance_id}/query/v2"


class GraphBuilder:
    """
    Loads extracted entities into Neo4j via HTTP Query API.

    Uses HTTPS (port 443) — works on all networks including those that
    block Bolt protocol port 7687.

    All Cypher queries use MERGE (not CREATE) so running multiple times
    is safe — no duplicate nodes or relationships are created.
    """

    def __init__(self) -> None:
        if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
            raise FinSightError(
                "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are all required. "
                "Check your .env file."
            )

        # Extract instance ID from username (they match in AuraDB)
        self.instance_id = NEO4J_USERNAME
        self.api_url = _build_http_url(NEO4J_URI, self.instance_id)
        self.auth = (NEO4J_USERNAME, NEO4J_PASSWORD)

        logger.info(
            "GraphBuilder initialized",
            extra={"api_url": self.api_url},
        )

        # Verify connection on startup
        self._verify_connection()

    def _run_query(self, cypher: str, parameters: Optional[dict] = None) -> dict:
        """
        Execute a Cypher query via the HTTP Query API.

        Args:
            cypher:     Cypher query string
            parameters: Optional query parameters dict

        Returns:
            API response as dict

        Raises:
            FinSightError: If the HTTP request fails or Neo4j returns an error
        """
        payload = {"statement": cypher}
        if parameters:
            payload["parameters"] = parameters

        try:
            response = requests.post(
                self.api_url,
                auth=self.auth,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except requests.exceptions.ConnectionError as e:
            raise FinSightError(
                f"Cannot reach Neo4j API at {self.api_url}. "
                f"Check your internet connection. Error: {e}"
            ) from e
        except requests.exceptions.Timeout:
            raise FinSightError(
                f"Neo4j API timed out at {self.api_url}. "
                "The instance may be starting up — wait 30 seconds and retry."
            )

        if response.status_code == 401:
            raise FinSightError(
                "Neo4j authentication failed (401). "
                "Check NEO4J_USERNAME and NEO4J_PASSWORD in .env."
            )
        if response.status_code == 404:
            raise FinSightError(
                f"Neo4j API endpoint not found (404): {self.api_url}. "
                "Check NEO4J_URI and NEO4J_USERNAME match your AuraDB instance."
            )
        if not response.ok:
            raise FinSightError(
                f"Neo4j API error {response.status_code}: {response.text[:200]}"
            )

        result = response.json()

        # Check for Cypher-level errors in the response body
        errors = result.get("errors", [])
        if errors:
            raise FinSightError(
                f"Cypher error: {errors[0].get('message', 'Unknown error')}"
            )

        return result

    def _verify_connection(self) -> None:
        """Test the connection with a simple query on startup."""
        try:
            self._run_query("RETURN 1 AS test")
            logger.info(
                f"Neo4j connection verified via HTTP API: {self.api_url}"
            )
            print(f"Neo4j connected via HTTP API: {self.api_url}")
        except FinSightError as e:
            raise FinSightError(
                f"Neo4j connection verification failed: {e}"
            ) from e

    def create_indexes(self) -> None:
        """Create indexes for faster entity lookups."""
        queries = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_company IF NOT EXISTS FOR (e:Entity) ON (e.source_company)",
        ]
        for q in queries:
            try:
                self._run_query(q)
            except FinSightError as e:
                # Index may already exist — not a fatal error
                logger.debug(f"Index creation note: {e}")
        logger.info("Neo4j indexes ensured")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships. USE WITH CAUTION."""
        self._run_query("MATCH (n) DETACH DELETE n")
        logger.warning("Neo4j graph cleared — all data deleted")
        print("Graph cleared.")

    def _merge_entity(
        self,
        entity: dict,
        source_company: str,
        doc_type: str,
        year: str,
    ) -> None:
        """Create or update an entity node. MERGE prevents duplicates."""
        name = entity.get("name", "").strip()
        etype = entity.get("type", "COMPANY").upper()

        if not name or len(name) < 2:
            return
        if etype not in VALID_ENTITY_TYPES:
            etype = "COMPANY"  # safe default

        cypher = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.source_company = $source_company,
            e.doc_type = $doc_type,
            e.year = $year
        """
        self._run_query(cypher, {
            "name": name,
            "type": etype,
            "source_company": source_company,
            "doc_type": doc_type,
            "year": year,
        })

    def _merge_relationship(
        self,
        rel: dict,
        source_company: str,
    ) -> None:
        """Create or update a relationship between two entity nodes."""
        from_name = rel.get("from", "").strip()
        to_name = rel.get("to", "").strip()
        relation = rel.get("relation", "").upper().replace(" ", "_")
        context = rel.get("context", "")[:200]  # cap context length

        if not from_name or not to_name or len(from_name) < 2 or len(to_name) < 2:
            return
        if relation not in VALID_RELATION_TYPES:
            relation = "RELATED_TO"

        cypher = """
        MERGE (a:Entity {name: $from_name})
        MERGE (b:Entity {name: $to_name})
        MERGE (a)-[r:RELATED_TO {type: $relation}]->(b)
        SET r.context = $context,
            r.source_company = $source_company
        """
        self._run_query(cypher, {
            "from_name": from_name,
            "to_name": to_name,
            "relation": relation,
            "context": context,
            "source_company": source_company,
        })

    def load_from_jsonl(
        self,
        entities_file: Optional[Path] = None,
        batch_size: int = 50,
    ) -> dict:
        """
        Load all extracted entities into Neo4j via HTTP API.

        Reads data/knowledge_graph/extracted_entities.jsonl line by line.
        Uses MERGE so safe to run multiple times — no duplicates created.
        Processes in batches and reports progress every batch_size records.

        Args:
            entities_file: Override path to JSONL file. Defaults to ENTITIES_FILE.
            batch_size:    How many records to process before printing progress.

        Returns:
            Summary dict: records_processed, entities_merged, relationships_merged
        """
        file_path = entities_file or ENTITIES_FILE

        if not file_path.exists():
            raise FinSightError(
                f"Entities file not found: {file_path}. "
                "Run: python scripts/extract_entities.py first."
            )

        self.create_indexes()

        # Count total lines for progress display
        with open(file_path, encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())

        print(f"\nLoading {total_lines} records into Neo4j via HTTP API...")
        print(f"API endpoint: {self.api_url}")

        records_processed = 0
        total_entities = 0
        total_relationships = 0
        errors = 0

        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line {i}")
                    continue

                company = record.get("company_name", "Unknown")
                doc_type = record.get("doc_type", "Unknown")
                year = record.get("year", "Unknown")

                # Merge entities
                for entity in record.get("entities", []):
                    try:
                        self._merge_entity(entity, company, doc_type, year)
                        total_entities += 1
                    except FinSightError as e:
                        logger.debug(f"Entity merge failed: {e}")
                        errors += 1

                # Merge relationships
                for rel in record.get("relationships", []):
                    try:
                        self._merge_relationship(rel, company)
                        total_relationships += 1
                    except FinSightError as e:
                        logger.debug(f"Relationship merge failed: {e}")
                        errors += 1

                records_processed += 1

                if i % batch_size == 0 or i == total_lines:
                    print(
                        f"  [{i}/{total_lines}] "
                        f"Entities: {total_entities} | "
                        f"Relationships: {total_relationships} | "
                        f"Errors: {errors}"
                    )

        summary = {
            "records_processed": records_processed,
            "total_entities_merged": total_entities,
            "total_relationships_merged": total_relationships,
            "errors": errors,
        }
        logger.info("Graph loading complete", extra=summary)

        print(f"\nNeo4j loading complete:")
        print(f"  Records processed:    {records_processed}")
        print(f"  Entities merged:      {total_entities}")
        print(f"  Relationships merged: {total_relationships}")
        print(f"  Errors skipped:       {errors}")
        return summary

    def get_stats(self) -> dict:
        """Return current count of nodes and relationships in the graph."""
        node_result = self._run_query("MATCH (n) RETURN count(n) AS count")
        rel_result = self._run_query("MATCH ()-[r]->() RETURN count(r) AS count")

        # Parse HTTP API response format
        node_count = node_result.get("data", {}).get("values", [[0]])[0][0]
        rel_count = rel_result.get("data", {}).get("values", [[0]])[0][0]

        return {"nodes": node_count, "relationships": rel_count}

    def run_custom_query(self, cypher: str) -> list:
        """
        Run any Cypher query and return results as list of dicts.
        Useful for testing and debugging the graph.

        Example:
            builder.run_custom_query("MATCH (n:Entity) RETURN n.name, n.type LIMIT 10")
        """
        result = self._run_query(cypher)
        data = result.get("data", {})
        fields = data.get("fields", [])
        values = data.get("values", [])
        return [dict(zip(fields, row)) for row in values]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("Testing Neo4j HTTP API connection...")
    try:
        builder = GraphBuilder()

        # Show current stats
        stats = builder.get_stats()
        print(f"Current graph: {stats}")

        # Test a simple query
        result = builder.run_custom_query(
            "MATCH (n:Entity) RETURN n.name, n.type LIMIT 5"
        )
        if result:
            print(f"Sample entities: {result}")
        else:
            print("Graph is empty — run python scripts/extract_entities.py first")

    except FinSightError as e:
        print(f"Connection failed: {e}")