from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Switch to alternative T5 model
model_name = "prithivida/parrot_paraphraser_on_T5"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=False)

def paraphrase_text(text):
    paraphraser = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.7
    )
    
    return paraphraser(text)[0]['generated_text']

if __name__ == "__main__":
    text = (
        "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. "
        "Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. "
        "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. "
        "I am very appreciated the full support of the professor, for our Springer proceedings publication."
    )
    
    try:
        result = paraphrase_text(text)
        print("Paraphrased:", result)
    except Exception as e:
        print(f"Error: {e}")
        print("Try alternative model: prithivida/parrot_paraphraser_on_T5")