from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_paraphrasing_pipeline():
    print("Loading Vennify grammar correction model...")
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def paraphrase(text, model, tokenizer):
    input_text = f"correct grammar: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def main():
    text = ("""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication
""")
    model, tokenizer = load_paraphrasing_pipeline()
    result = paraphrase(text, model, tokenizer)
    print("Original:", text)
    print("Corrected:", result)

if __name__ == "__main__":
    main()
