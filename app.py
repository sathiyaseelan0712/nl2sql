from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging  
logging.basicConfig(level=logging.DEBUG)  
tokenizer = AutoTokenizer.from_pretrained("suriya7/t5-base-text-to-sql")
model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/t5-base-text-to-sql")
app = Flask(__name__)
@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]
        sql_query = translate_to_sql_select(msg)
        logging.debug(f"SQL query: {sql_query}")
        return jsonify({"response": sql_query})
def translate_to_sql_select(english_query):
    input_text = "translate English to SQL: " + english_query
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(input_ids)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query.strip()
if __name__ == '__main__':
    app.run(debug=True)