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
    text = ("Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. "
            "Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. "
            "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. "
            "I am very appreciated the full support of the professor, for our Springer proceedings publication.")
    
    paraphrases = generate_paraphrases(text)
    # Print only the last paraphrase
    print(f"Paraphrase:\n{paraphrases[-1]}\n")
