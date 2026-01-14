# Extracting SPOs.
# python -m spacy download en -> To donwload the en model of spacy.

import spacy
import textacy


# Subject Verb Object detection


class KnowledgeExtraction:

    def retrieveKnowledge(self, textInput):
        nlp = spacy.load('en')
        text = nlp(textInput)
        text_ext = textacy.extract.subject_verb_object_triples(text)
        return list(text_ext)


def generate_random_password(length):
    # Define the characters to use for the password
    characters = string.ascii_letters + string.digits  # Alphanumeric characters
    
    # Generate a random password of the specified length
    password = ''.join(random.choice(characters) for _ in range(length))
    
    return password