import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/processed.txt',
    model_prefix='tokenizer',
    vocab_size=50000,
    user_defined_symbols=['apq', 'rst'],
    character_coverage=1.0,
    model_type="bpe",
)