from app.core.config import get_settings, get_qdrant_client

settings = get_settings()
client = get_qdrant_client(settings)

FORBIDDEN = [
    "manual fraud review",
    "extension 4410",
    "risk team",
    "orders above rs 50,000",
]

try:
    info = client.get_collection(settings.collection_name)

    print("Collection:", settings.collection_name)
    print("Total points:", info.points_count)

    offset = None
    matches = []

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            content = str(payload).lower()

            for phrase in FORBIDDEN:
                if phrase in content:
                    matches.append((record.id, phrase))

        if next_offset is None:
            break

        offset = next_offset

    print("\n========== SECURITY AUDIT ==========")

    if matches:
        print("SECURITY FAILURE: Restricted content found!")

        for record_id, phrase in matches:
            print("Point:", record_id)
            print("Restricted phrase:", phrase)
    else:
        print("PASS: No forbidden phrases found in indexed payloads.")

finally:
    client.close()