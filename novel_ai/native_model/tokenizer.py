"""Fixed byte tokenizer. Own implementation, no pretrained vocabulary needed."""
class ByteTokenizer:
    vocab_size = 260
    pad_id = 256
    eos_id = 257
    bos_id = 258
    sep_id = 259
    def encode(self, text, add_bos=False, add_eos=False):
        return ([self.bos_id] if add_bos else []) + list(text.encode('utf-8')) + ([self.eos_id] if add_eos else [])
    def decode(self, ids):
        return bytes(i for i in ids if 0 <= int(i) < 256).decode('utf-8', errors='replace')
