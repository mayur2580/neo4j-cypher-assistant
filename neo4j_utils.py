from neo4j import GraphDatabase

def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


def test_connection(uri: str, user: str, password: str):
    driver = get_driver(uri, user, password)
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)
    finally:
        driver.close()


def get_schema(uri: str, user: str, password: str, database: str):
    driver = get_driver(uri, user, password)

    schema = {
        "labels": [],
        "relationships": [],
        "properties": []
    }

    try:
        with driver.session(database=database) as session:

            schema["labels"] = [
                r["label"]
                for r in session.run(
                    "CALL db.labels() YIELD label RETURN label"
                )
            ]

            schema["relationships"] = [
                r["relationshipType"]
                for r in session.run(
                    """
                    CALL db.relationshipTypes()
                    YIELD relationshipType
                    RETURN relationshipType
                    """
                )
            ]

            schema["properties"] = [
                r["propertyKey"]
                for r in session.run(
                    """
                    CALL db.propertyKeys()
                    YIELD propertyKey
                    RETURN propertyKey
                    """
                )
            ]

        return schema

    finally:
        driver.close()


def run_cypher(
    uri: str,
    user: str,
    password: str,
    database: str,
    cypher: str
):
    driver = get_driver(uri, user, password)

    results = []

    try:
        with driver.session(database=database) as session:
            data = session.run(cypher)

            for row in data:
                results.append(dict(row))

        return results

    finally:
        driver.close()