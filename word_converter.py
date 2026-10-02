from encoder import encode, decode

random_word = '*tasty$______'
encoded = encode(random_word)
print(encoded)

word = decode(encoded)
print(word)