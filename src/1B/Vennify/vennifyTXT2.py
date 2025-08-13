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
    text = ("""During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets
""")
    model, tokenizer = load_paraphrasing_pipeline()
    result = paraphrase(text, model, tokenizer)
    print("Original:", text)
    print("Corrected:", result)

if __name__ == "__main__":
    main()
