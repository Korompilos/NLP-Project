import random

def reconstruct_sentence(sentence):
    replacements = {
        "I believe the team": [
            "Despite some challenges, I trust the team",
            "I am confident that the team",
            "It is clear the team"
        ],
        "although bit delay and less communication at recent days": [
            "even though there have been some delays and reduced communication recently",
            "despite recent slow progress and less interaction",
            "although communication and progress were a bit slow lately"
        ],
        "they really tried best for paper and cooperation": [
            "they have truly done their best with the paper and teamwork",
            "they genuinely gave their best effort on the manuscript and collaboration",
            "they worked hard and cooperated well on the paper"
        ]
    }
    
    for old, options in replacements.items():
        if old in sentence:
            sentence = sentence.replace(old, random.choice(options))
    
    return sentence

# Δοκιμή
sentence2 = "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

print(reconstruct_sentence(sentence2))