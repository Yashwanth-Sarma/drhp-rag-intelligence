from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print("URI:", uri)
print("USER:", user)

driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
)

driver.verify_connectivity()
print("CONNECTED!")

with driver.session() as session:
    result = session.run("RETURN 1 AS x")
    print(result.single()["x"])

driver.close()