# docs/

Design notes and the two wiring guides.

| Document | Read it when |
|---|---|
| `architecture.md` | you want the shape of the whole thing: the two boundaries, the review harness, error blast radius, durability, and why the model layer is a "backend" and not an "adapter" |
| `languages.md` | you are setting up a language pair — same-script vs different-script detection, adding a language, and per-language concerns (register, transliteration, pro-drop, RTL, CJK width) |
| `reference-translations.md` | you want to match an existing body of translation — an external reference corpus retrieved per unit (translation memory / RAG), a store that can learn from the run's own output, and the optional embedding / hybrid-fusion / cross-encoder-reranking retrievers (with what they measured on real corpora — including where hybrid fusion does *not* pay off, why robustness on your own corpus is the only measured reason to prefer one retriever over another, and a retracted consistency finding kept on the page as a worked example of a circular metric) |
| `writing-a-carrier-adapter.md` | you are connecting a source of text (subtitles, a game, documents) to the translator — the downstream boundary |
| `writing-a-backend.md` | you are adding support for a new local model — the upstream boundary |

For the runnable specifics, the package READMEs are closer to the code: `src/transunit/`,
`src/translator/`, `src/translator/backend/`, and `config/`.
