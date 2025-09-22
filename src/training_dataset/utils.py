from typing import List


def split_text_into_chunks_with_offsets(
    text: str, max_chunk_size: int = 4000, overlap: int = 100
) -> List[tuple[str, int, int]]:
    """Split text into manageable chunks with character offsets and overlap."""
    if len(text) <= max_chunk_size:
        return [(text, 0, len(text))]

    chunks_with_offsets = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))

        # If not at the end, try to find a good break point (sentence boundary)
        if end < len(text):
            # Look for sentence boundaries within the last 200 characters
            search_start = max(start, end - 200)
            sentence_end = text.rfind(". ", search_start, end)
            if sentence_end != -1:
                end = sentence_end + 2  # Include the period and space

        chunk = text[start:end]
        chunks_with_offsets.append((chunk, start, end))

        # If we've reached the end, break
        if end >= len(text):
            break

        # Move start forward, accounting for overlap
        start = max(start + 1, end - overlap)

    return chunks_with_offsets


def split_text_into_chunks(
    text: str, max_chunk_size: int = 4000, overlap: int = 100
) -> List[str]:
    """Split text into manageable chunks for processing."""
    chunks_with_offsets = split_text_into_chunks_with_offsets(
        text, max_chunk_size, overlap
    )
    return [chunk for chunk, _, _ in chunks_with_offsets]

