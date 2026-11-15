from app.core.config import get_settings, get_qdrant_client

settings = get_settings()
client = get_qdrant_client(settings)

name = settings.collection_name

try:
    print("========== BEFORE DELETION ==========")

    print("Collections:", client.get_collections().collections)

    if client.collection_exists(name):
        info = client.get_collection(name)
        print("Collection:", name)
        print("Points:", info.points_count)

        result = client.delete_collection(name)
        print("Delete result:", result)

    print("\n========== AFTER DELETION ==========")

    print("Collection exists:", client.collection_exists(name))
    print("Collections:", client.get_collections().collections)

    print("\n========== RECREATING ==========")

    from qdrant_client.models import Distance, VectorParams

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.embed_dim,
            distance=Distance.COSINE,
        ),
    )

    info = client.get_collection(name)

    print("New collection created:", name)
    print("Points immediately after creation:", info.points_count)

    print("\n========== FINAL CHECK ==========")
    print(
        "Exact point count:",
        client.count(collection_name=name, exact=True).count,
    )

finally:
    client.close()