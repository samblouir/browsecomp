from browsecomp250.crypto import decrypt, encrypt


def test_crypto_round_trip() -> None:
    canary = "browsecomp:test-canary"
    plaintext = "A difficult BrowseComp question?"
    assert decrypt(encrypt(plaintext, canary), canary) == plaintext
