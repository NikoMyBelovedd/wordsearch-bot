# words.txt

~284,000 uppercase words, 3+ letters, **frequency-ordered**: the bot fires them top-down,
so common words are tried before obscure ones. Rebuild with `tools/build_words.py`.

Sources, in priority order:
1. [google-10000-english](https://github.com/first20hours/google-10000-english): the first 10,000 lines
2. [SCOWL](http://wordlist.aspell.net/) 2020.12.07, size levels 10 to 80 (words, spelling variants, capitalised
   words and proper names; English, American, British, Canadian, Australian), by level.
   Copyright 2000-2018 Kevin Atkinson; licence and notices in `SCOWL-LICENSE.txt`.

Theme words outside it (brands, run-together phrases: ZUMBA, JETSKI, ALOEVERA) are found
by the exhaustive pass and then learned into `local/learned-words.txt`.
