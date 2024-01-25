import re
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder(SentenceTransformer):
    """
    Builds on top of SentenceTransformer from the sentence_transformers module.
    Implements `unlimited_length_encoding`, avoiding truncation of long texts.
    """

    # TODO: Use a sliding window approach to split up long texts into chunks so less context is lost.
    def embedd(
        self, texts, allow_truncation=False, **kwargs
    ) -> np.array:
        """
        Avoid truncation of long texts by encoding long texts segment-by-segment
        and aggregating the resulting vectors, weighing them by the word lengths.
        """
        # If truncation of long texts is okay, we simply call `SentenceTransformer.encode``
        # which truncates texts that are too long (e.g., over 256 tokens).
        if allow_truncation:
            return self.encode(texts, **kwargs)
        # We divide by 1.5 since a word is, on average, longer than one token.
        # This division has the advantage that text is less often "lost" to the embedding.
        # However, it has the disadvantage that context is lost when a chunk could have been longer
        # but was split up into two independent embeddings.
        max_sequence_length = round((self.get_max_seq_length() or 256) / 1.5)
        embedding_dimension = self.get_sentence_embedding_dimension() or 512
        # Iterate over all texts, split them into segments of appropriate lengths
        # and encode them one-by-one, aggregating the resulting vectors in the end.
        # TODO: Optimize this using the fact that one may enter batches
        result = np.zeros((len(texts), embedding_dimension))
        for i, text in enumerate(texts):
            split_text = self._split_text(text, max_sequence_length)
            embedding_parts = np.zeros((len(split_text), embedding_dimension))
            weights = np.zeros(shape=len(split_text))
            for j, (text_segment, segment_length) in enumerate(split_text):
                new_embedding_part = self.encode(
                    sentences=text_segment, convert_to_numpy=True, **kwargs
                )
                # We don't need `new_embedding_part[0]` as long as we pass a string to `encode`.
                embedding_parts[j] = new_embedding_part
                weights[j] = segment_length
            # Normalize the weights so the weighted sum doesn't scale the result.
            weights = weights / weights.sum()
            embedding_parts = np.array(embedding_parts)
            result[i] = np.dot(weights, embedding_parts)
        return result

    @staticmethod
    def _split_text(text: str, max_sequence_length: int) -> list[tuple[str, int]]:
        """
        Splits a string into chunks of a maximum number of words
        while respecting original white spaces.

        Parameters
        ----------
        text (str):
            The string to be split.
        max_length (int):
            The maximum number of words in each chunk.

        Returns
        -------
        result
            A list of tuples, where the first element of each tuple is a string chunk,
            and the second element is the word count in the respective chunk.
        """
        # If the total number of words in the string is less than or equal to max_length
        if len(text.split()) <= max_sequence_length:
            return [(text, len(text.split()))]

        chunks = re.findall(R"\S+|\s+", text)  # Split the string into words and spaces

        result = []
        current_chunk = ""
        current_length = 0

        for chunk in chunks:
            # Check if the chunk is a word
            is_word = chunk.strip() != ""

            # If the chunk is a word and adding it won't exceed the maximum length
            if is_word and current_length + 1 <= max_sequence_length:
                current_chunk += chunk
                current_length += 1
            # If the chunk is a space, just add it to the current chunk
            elif not is_word:
                current_chunk += chunk
            else:
                # Otherwise, store the current chunk and its word count in the result
                result.append((current_chunk, current_length))
                # Start a new chunk with the current word
                current_chunk = chunk
                current_length = 1 if is_word else 0

        # If there are any words left in the current chunk, add it to the result
        if current_chunk:
            result.append((current_chunk, current_length))

        return result
