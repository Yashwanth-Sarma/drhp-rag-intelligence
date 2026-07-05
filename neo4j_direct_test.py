from neo4j import GraphDatabase

URI = "bolt+s://4660f290.databases.neo4j.io:7687"
USER = "neo4j"
PASSWORD = "49f6KYiZ3NrQUytu4sF7mFpRXs2knF7FyqM0NXBv-cs"

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD),
)

print("Connecting...")

with driver.session(database="neo4j") as session:
    result = session.run("RETURN 1 AS x")
    print(result.single()["x"])

driver.close()

print("SUCCESS")