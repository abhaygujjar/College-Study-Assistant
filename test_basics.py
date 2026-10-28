import numpy as np
import pytest

from text_chunker import Chunk, chunk_pages, split_words
from pdf_loader import PageText, load_document
from vector_store import VectorStore


def test_split_words_overlap():
    text = " ".join(str(i) for i in range(100))
    pieces = split_words(text, chunk_size=30, overlap=10)
    assert pieces[0].split()[0] == "0"
    # last 10 words of piece 1 should start piece 2
    assert pieces[0].split()[-10:] == pieces[1].split()[:10]
    # every word must appear somewhere
    seen = set(" ".join(pieces).split())
    assert seen == set(str(i) for i in range(100))


def test_split_words_short_text_is_one_chunk():
    assert split_words("hello world", 50, 10) == ["hello world"]


def test_split_words_empty():
    assert split_words("   ", 50, 10) == []


def test_split_words_bad_settings():
    with pytest.raises(ValueError):
        split_words("a b c", chunk_size=10, overlap=10)


def test_chunk_pages_keeps_source_and_page():
    pages = [PageText("notes.pdf", 3, "word " * 100)]
    chunks = chunk_pages(pages, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    assert all(c.source == "notes.pdf" and c.page == 3 for c in chunks)


def test_load_text_file():
    pages = load_document(b"Stack is LIFO.\n\nQueue is FIFO.", "ds.txt")
    assert len(pages) == 1
    assert "LIFO" in pages[0].text


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        load_document(b"data", "slides.pptx")


def test_vector_store_finds_closest():
    store = VectorStore(dim=3)
    chunks = [Chunk("a", "f", 1), Chunk("b", "f", 2), Chunk("c", "g", 1)]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    store.add(chunks, vectors)

    hits = store.search(np.array([0, 1, 0], dtype="float32"), k=1)
    assert hits[0][0].text == "b"


def test_vector_store_remove_source():
    store = VectorStore(dim=2)
    chunks = [Chunk("a", "f", 1), Chunk("b", "g", 1)]
    store.add(chunks, np.array([[1, 0], [0, 1]], dtype="float32"))
    store.remove_source("f")
    assert len(store) == 1
    assert store.search(np.array([0, 1], dtype="float32"), k=1)[0][0].text == "b"


def test_reindexing_does_not_duplicate_chunks(monkeypatch):
    import rag_pipeline as rag

    monkeypatch.setattr(
        rag, "embed_texts", lambda texts: np.random.rand(len(texts), 384).astype("float32")
    )
    assistant = rag.StudyAssistant()
    data = ("SQL views are saved queries. " * 200).encode()

    first = assistant.add_document(data, "sql.txt")
    assistant.files.clear()  # pretend a past run was interrupted
    assistant.add_document(data, "sql.txt")

    assert len(assistant.store) == first


def test_llm_finds_new_model_after_404(monkeypatch):
    import llm_client as llm

    class Resp:
        def __init__(self, code, data=None, text=""):
            self.status_code, self._data, self.text = code, data or {}, text

        def json(self):
            return self._data

    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if "gemini-9.9-flash" in url:
            return Resp(200, {"candidates": [{"content": {"parts": [{"text": "ok answer"}]}}]})
        return Resp(404, text="model removed")

    def fake_get(url, **kwargs):
        models = [
            {"name": "models/gemini-3.0-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-9.9-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-9.9-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-12-flash", "supportedGenerationMethods": ["embedContent"]},
        ]
        return Resp(200, {"models": models})

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "_discovered_model", None)
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.requests, "get", fake_get)

    hits = [(Chunk("x", "f.pdf", 1), 0.9)]
    assert llm.ask_llm("what is x?", hits) == "ok answer"
    assert len(calls) == 2  # first the old name (404), then the discovered one


def test_keyword_overlap_matches_word_forms():
    from rag_pipeline import keyword_overlap

    assert keyword_overlap("what's VIEW?", "11. Views Simple explanation A view is a saved query") == 1.0
    assert keyword_overlap("what's VIEW?", "CTEs are temporary result sets") == 0.0
    assert keyword_overlap("what is the", "anything") == 0.0  # only stopwords


def test_rerank_prefers_chunk_with_the_keyword():
    from rag_pipeline import rerank

    index_chunk = Chunk("CREATE INDEX speeds up searches", "f", 6)
    view_chunk = Chunk("A view is a saved query", "f", 7)
    # the meaning score alone prefers the wrong chunk (0.12 vs 0.09)
    ranked = rerank("what's VIEW?", [(index_chunk, 0.12), (view_chunk, 0.09)])
    assert ranked[0][0].page == 7


def test_llm_retries_when_google_is_busy(monkeypatch):
    import llm_client as llm

    class Resp:
        def __init__(self, code, data=None):
            self.status_code, self._data, self.text = code, data or {}, "busy"

        def json(self):
            return self._data

    answers = iter([Resp(503), Resp(503), Resp(200, {"candidates": [{"content": {"parts": [{"text": "done"}]}}]})])
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "_discovered_model", None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: next(answers))

    assert llm.ask_llm("q", [(Chunk("x", "f", 1), 0.9)]) == "done"


def test_llm_uses_backup_model_if_main_stays_busy(monkeypatch):
    import llm_client as llm

    class Resp:
        def __init__(self, code, data=None):
            self.status_code, self._data, self.text = code, data or {}, "busy"

        def json(self):
            return self._data

    def fake_post(url, **kw):
        if "flash-lite" in url:
            return Resp(200, {"candidates": [{"content": {"parts": [{"text": "from backup"}]}}]})
        return Resp(503)

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "_discovered_model", None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.requests, "post", fake_post)

    assert llm.ask_llm("q", [(Chunk("x", "f", 1), 0.9)]) == "from backup"


def test_llm_retries_after_timeout(monkeypatch):
    import llm_client as llm

    class Resp:
        status_code, text = 200, ""

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "after timeout"}]}}]}

    calls = {"n": 0}

    def fake_post(url, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise llm.requests.exceptions.ReadTimeout("slow")
        return Resp()

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "_discovered_model", None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.requests, "post", fake_post)

    assert llm.ask_llm("q", [(Chunk("x", "f", 1), 0.9)]) == "after timeout"


def test_llm_timeout_every_time_gives_clear_error(monkeypatch):
    import llm_client as llm

    def always_timeout(url, **kw):
        raise llm.requests.exceptions.ReadTimeout("slow")

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "_discovered_model", None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.requests, "post", always_timeout)

    with pytest.raises(llm.LLMError, match="timeout"):
        llm.ask_llm("q", [(Chunk("x", "f", 1), 0.9)])


def test_empty_llm_answer_is_reported(monkeypatch):
    import llm_client as llm

    class Resp:
        status_code, text = 200, ""

        def json(self):
            return {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "_discovered_model", None)
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Resp())

    with pytest.raises(llm.LLMError, match="MAX_TOKENS"):
        llm.ask_llm("q", [(Chunk("x", "f", 1), 0.9)])


def test_model_loading_disarms_torch_classes_watcher_bug(monkeypatch):
    import sys
    import types

    import embedder

    fake_torch = types.SimpleNamespace(classes=types.SimpleNamespace())

    class FakeModel:
        def __init__(self, name, device=None):
            self.max_seq_length = 0

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel)
    )
    embedder._load_model.cache_clear()
    try:
        model = embedder._load_model()
        assert fake_torch.classes.__path__ == []
        assert model.max_seq_length == 256
    finally:
        embedder._load_model.cache_clear()


def test_out_of_memory_gives_clear_error(monkeypatch):
    import embedder

    def fake_load():
        raise OSError("The paging file is too small for this operation to complete. (os error 1455)")

    monkeypatch.setattr(embedder, "_load_model", fake_load)
    with pytest.raises(embedder.ModelLoadError, match="ran out of memory"):
        embedder.warm_up()


def test_other_os_errors_are_not_hidden(monkeypatch):
    import embedder

    monkeypatch.setattr(embedder, "_load_model", lambda: (_ for _ in ()).throw(OSError("disk broken")))
    with pytest.raises(OSError, match="disk broken"):
        embedder.warm_up()
