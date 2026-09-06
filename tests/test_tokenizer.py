from llmz.tokenizer import BYTE_OFFSET, BytesTokenizer, PairTokenizer


def test_bytes_round_trip():
    tokenizer = BytesTokenizer()
    text = "lea (%rdi), %eax  // café\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_pair_namespaces_do_not_overlap():
    tokenizer = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    source = tokenizer.encode_source("a")
    target = tokenizer.encode_target("a")
    assert source[0] >= BYTE_OFFSET
    assert target[0] >= tokenizer.source.vocab_size
    assert tokenizer.decode_source(source) == "a"
    assert tokenizer.decode_target(target) == "a"
