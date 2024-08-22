def normalize_distances(distances):
    min_dist = min(distances)
    max_dist = max(distances)
    return [(dist - min_dist) / (max_dist - min_dist) for dist in distances]


def combine_faiss_results(faiss_index_kb1, faiss_index_kb2, query_vector, top_n=10):
    # Step 1: Query both knowledge bases
    distances_kb1, indices_kb1 = faiss_index_kb1.search(query_vector, top_n)
    distances_kb2, indices_kb2 = faiss_index_kb2.search(query_vector, top_n)

    # Step 2: (Optional) Normalize distances
    distances_kb1 = normalize_distances(distances_kb1)
    distances_kb2 = normalize_distances(distances_kb2)

    # Step 3: Merge and sort results
    combined_results = list(zip(distances_kb1, indices_kb1)) + list(zip(distances_kb2, indices_kb2))
    combined_results.sort(key=lambda x: x[0])  # Sort by normalized distance

    # Step 4: Select top N results
    final_results = combined_results[:top_n]

    return final_results
