from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """
During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets
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
