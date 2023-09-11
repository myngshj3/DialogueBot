from dialogue_bot import DialogueBot
from flask import Flask, request, jsonify

app = Flask(__name__)
bot = DialogueBot()

@app.route('/generate', methods=['POST'])
def generate():
    global bot
    text = request.json['text']
    generated_text = bot.have_dialogue(text)
    return jsonify({'generated_text': generated_text})

def main():
    global app, bot
    bot.configure()
    bot.load_model()
    print("Bot setup done!")
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()