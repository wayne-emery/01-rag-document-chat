# Example Documents

Drop any of these into the upload panel to try the system without preparing your own files.

| File | Good for testing |
|---|---|
| `paris_landmarks.txt` | Multi-source retrieval — questions like _"Which Paris landmarks were completed in the 1800s?"_ force the system to pull and cite multiple chunks across different landmarks. |
| `rag_intro.txt` | Self-referential demo — ask _"Why use RAG instead of fine-tuning?"_ and watch it cite the relevant paragraph from this very file. |
| `grace_hopper.txt` | Single-document factual recall — _"What was the first compiler Grace Hopper wrote and when?"_ |

## Suggested questions for stress-testing

**Refusal behavior** (should answer: "I don't have enough information..."):
- After uploading only `grace_hopper.txt`, ask _"Who designed the Eiffel Tower?"_

**Multi-source synthesis**:
- Upload all three, then ask _"What do these documents have in common?"_

**Date arithmetic** (tests grounding, not math):
- _"How old was Grace Hopper when she retired from the Navy?"_ — answer is in the text, not derived.

**Specific citation**:
- _"Which year was the term 'debugging' popularized?"_ — should cite the chunk about the moth in the Mark II relay.
