from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """
Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication 
"""

inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=120,
    min_length=60,
    no_repeat_ngram_size=2,
    length_penalty=1.0,
    early_stopping=True,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
