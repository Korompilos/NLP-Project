from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Ateeqq/Text-Rewriter-Paraphraser")
model = AutoModelForSeq2SeqLM.from_pretrained("Ateeqq/Text-Rewriter-Paraphraser")

def generate_paraphrases(text, num_return_sequences=3):
    # Add prefix as expected by the model
    input_text = "paraphraser: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    # Generate paraphrases using beam search
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=6,
        num_return_sequences=num_return_sequences,
        num_beam_groups=3,
        diversity_penalty=2.0,
        early_stopping=True,
        trust_remote_code=True
    )
    
    # Decode and return paraphrased texts
    paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return paraphrases

if __name__ == "__main__":
    text = ("""During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets""")
    
    paraphrases = generate_paraphrases(text)
    # Print only the last paraphrase
    print(f"Paraphrase:\n{paraphrases[-1]}\n")
