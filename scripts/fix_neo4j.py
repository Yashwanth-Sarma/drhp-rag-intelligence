"""
scripts/fix_neo4j.py

Diagnoses and fixes Neo4j AuraDB SSL connection issues on Windows.
The root cause: Python's certifi bundle is missing SSL.com intermediate
certificates that AuraDB free tier uses. Browser works because it has
its own certificate store. Python does not.

Run: python scripts/fix_neo4j.py
"""

import sys
import os
import ssl
import socket
import certifi

# ── Step 1: Download and append missing SSL.com intermediate certs ─────────
SSL_COM_INTERMEDIATES = [
    # SSL.com RSA SSL subCA — used by Neo4j AuraDB free tier
    "https://crt.sh/?d=3537113745",
    # SSL.com EV RSA SSL CA — backup intermediate
    "https://crt.sh/?d=6130972",
]

def download_and_append_cert(url: str, label: str) -> bool:
    """Download a certificate and append it to certifi's bundle."""
    try:
        import urllib.request
        import tempfile

        print(f"Downloading: {label}")
        req = urllib.request.urlopen(url, timeout=10)
        cert_data = req.read()

        # Decode DER to PEM if needed
        if cert_data.startswith(b"\x30"):  # DER format starts with 0x30
            import base64
            pem = (
                b"-----BEGIN CERTIFICATE-----\n"
                + base64.encodebytes(cert_data)
                + b"-----END CERTIFICATE-----\n"
            )
        else:
            pem = cert_data

        # Append to certifi bundle
        ca_bundle = certifi.where()
        with open(ca_bundle, "ab") as f:
            f.write(b"\n# " + label.encode() + b"\n")
            f.write(pem)

        print(f"  Appended to: {ca_bundle}")
        return True

    except Exception as e:
        print(f"  Failed to download {label}: {e}")
        return False


# ── Step 2: Test SSL handshake on Bolt port 7687 ──────────────────────────
def test_bolt_ssl(host: str) -> tuple[bool, str]:
    """Test raw SSL connection to Neo4j's Bolt port."""
    port = 7687
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                issued_to = dict(x[0] for x in cert.get("subject", []))
                issuer = dict(x[0] for x in cert.get("issuer", []))
                return True, f"OK — issued to {issued_to} by {issuer}"
    except ssl.SSLCertVerificationError as e:
        return False, f"SSL cert failed: {e}"
    except ConnectionRefusedError:
        return False, f"Connection refused on port {port} — check instance is running"
    except socket.timeout:
        return False, f"Timeout connecting to {host}:{port} — check firewall"
    except Exception as e:
        return False, f"Error: {e}"


# ── Step 3: Test full Neo4j driver connection ──────────────────────────────
def test_neo4j_driver(uri: str, user: str, password: str) -> tuple[bool, str]:
    """Test Neo4j Python driver connection."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=30,
            connection_timeout=15,
        )
        driver.verify_connectivity()
        driver.close()
        return True, "Neo4j driver connected successfully"
    except Exception as e:
        return False, str(e)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

    if not NEO4J_URI:
        print("ERROR: NEO4J_URI not found in .env")
        sys.exit(1)

    # Extract hostname from URI
    host = NEO4J_URI.replace("neo4j+s://", "").replace("neo4j://", "").rstrip("/")
    print(f"\nDiagnosing Neo4j connection to: {host}")
    print(f"certifi bundle: {certifi.where()}\n")

    # Test 1: SSL before fix
    print("=== Test 1: SSL handshake (before fix) ===")
    ok, msg = test_bolt_ssl(host)
    print(f"  Result: {'PASS' if ok else 'FAIL'} — {msg}")

    if not ok and "cert" in msg.lower():
        print("\n=== Applying SSL certificate fix ===")
        labels = ["SSL.com RSA SSL subCA", "SSL.com EV RSA SSL CA"]
        for url, label in zip(SSL_COM_INTERMEDIATES, labels):
            download_and_append_cert(url, label)

        # Test 2: SSL after fix
        print("\n=== Test 2: SSL handshake (after fix) ===")
        ok2, msg2 = test_bolt_ssl(host)
        print(f"  Result: {'PASS' if ok2 else 'FAIL'} — {msg2}")

    # Test 3: Full driver test
    print("\n=== Test 3: Neo4j driver connection ===")
    ok3, msg3 = test_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    print(f"  Result: {'PASS' if ok3 else 'FAIL'} — {msg3}")

    if ok3:
        print("\n✓ Neo4j connection working. You can now run:")
        print("  python scripts/build_graph.py")
    else:
        # Test 4: Neo4j 6.x compatible SSL bypass
        print("\n=== Test 4: Attempting with SSL verification disabled (neo4j 6.x) ===")
        print("  WARNING: Development only.")
    try:
        from neo4j import GraphDatabase, TrustAll

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASS),
            trusted_certificates=TrustAll(),
        )
        driver.verify_connectivity()
        driver.close()
        print("  PASS — connected with TrustAll()")
        print("\n  Add NEO4J_TRUST_ALL_CERTS=true to .env")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 5: Try neo4j:// instead of neo4j+s://
    print("\n=== Test 5: Trying plain bolt URI ===")
    bolt_uri = NEO4J_URI.replace("neo4j+s://", "bolt+s://")
    print(f"  URI: {bolt_uri}")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            bolt_uri,
            auth=(NEO4J_USER, NEO4J_PASS),
            max_connection_lifetime=30,
            connection_timeout=15,
        )
        driver.verify_connectivity()
        driver.close()
        print("  PASS — bolt+s:// works")
        print(f"\n  Update NEO4J_URI in .env to: {bolt_uri}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 6: Check what's in .env exactly
    print("\n=== Test 6: Credential check ===")
    print(f"  URI in .env:      '{NEO4J_URI}'")
    print(f"  Username in .env: '{NEO4J_USER}'")
    print(f"  Password length:  {len(NEO4J_PASS)} chars")
    print(f"  URI starts with:  '{NEO4J_URI[:15]}...'")
    expected_prefix = "neo4j+s://"
    if not NEO4J_URI.startswith(expected_prefix):
        print(f"  WARNING: URI should start with '{expected_prefix}'")
    if len(NEO4J_PASS) < 8:
        print("  WARNING: Password seems too short — verify it's copied correctly")