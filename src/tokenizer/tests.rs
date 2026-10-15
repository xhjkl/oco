use super::*;

#[test]
fn decoder_preserves_split_utf8_controls_and_invalid_byte_boundaries() -> Result<()> {
    let mut vocab: Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), id as u32))
        .collect();
    vocab.insert("<|stop|>".to_owned(), 256);
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .map_err(anyhow::Error::from_boxed)?;
    let mut encoder = tokenizers::Tokenizer::new(model);
    encoder.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    let mut kinds = vec![TokenKind::Normal; 256];
    kinds.push(TokenKind::Control);
    let tokenizer = Tokenizer {
        encoder,
        kinds,
        bos: None,
        eos: Some(256),
    };
    for text in ["", "ASCII\0\r\n", "ქართული café 你好 🤍✨ �"] {
        let ids = tokenizer.encode(text)?;
        assert_eq!(tokenizer.decode(&ids)?, text);
        let mut decoder = tokenizer.decoder();
        let mut output = String::new();
        for id in ids {
            output.push_str(&decoder.push(id)?);
            assert!(
                text.starts_with(&output),
                "partial character emitted prematurely"
            );
            assert!(decoder.pending.len() <= 3);
        }
        output.push_str(&decoder.finish());
        assert_eq!(output, text);
    }

    let lead = tokenizer.encoder.token_to_id("â").unwrap(); // UTF-8 lead byte E2.
    let ascii = tokenizer.encoder.token_to_id("A").unwrap();
    let mut decoder = tokenizer.decoder();
    assert_eq!(decoder.push(lead)?, "");
    assert!(decoder.push(u32::MAX).is_err());
    assert_eq!(decoder.push(ascii)?, "�A");
    assert_eq!(decoder.push(lead)?, "");
    assert_eq!(decoder.push(256)?, "�<|stop|>");
    assert_eq!(decoder.push(lead)?, "");
    assert_eq!(decoder.finish(), "�");
    assert!(tokenizer.decode(&[u32::MAX]).is_err());
    assert_eq!(tokenizer.special_token("<|stop|>"), Some(256));
    assert_eq!(tokenizer.special_token("A"), None);
    let literal = tokenizer.encode("<|stop|>")?;
    assert!(!literal.contains(&256));
    assert_eq!(tokenizer.decode(&literal)?, "<|stop|>");
    Ok(())
}
