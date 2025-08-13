import random

def reconstruct_sentence(sentence):
    replacements = {
        "Today is": [
            "Today we celebrate",
            "This day marks",
            "It is the day of"
        ],
        "in our Chinese culture": [
            "as part of our Chinese tradition",
            "within our rich Chinese heritage",
            "according to our Chinese customs"
        ],
        "to celebrate it with all safe and great in our lives": [
            "to enjoy it while ensuring everything in our lives is safe and wonderful",
            "to celebrate it while making sure all aspects of our lives are safe and great",
            "to honor it while keeping everything in our lives secure and positive"
        ]
    }
    
    for old, options in replacements.items():
        if old in sentence:
            sentence = sentence.replace(old, random.choice(options))
    
    return sentence

# Δοκιμή
sentence = "Today is our dragon boat festival, in our Chinese culture , to celebrate it with all safe and great in our lives."

print(reconstruct_sentence(sentence))