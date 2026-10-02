special_chars = ['*', '$', '_']

alphabets = [chr(letter) for letter in range(ord('a'), ord('z')+1)]

alphabets += special_chars

char_to_int = dict()
int_to_char = dict()

for letter in alphabets:
    char_to_int[letter] = alphabets.index(letter)
    int_to_char[alphabets.index(letter)] = letter

def encode(word):
    one_hot = [char_to_int[i] for i in word]
    return one_hot

def decode(tokens):
    word = [int_to_char[i] for i in tokens]
    return(''.join(word))






