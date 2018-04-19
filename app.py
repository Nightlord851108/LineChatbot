from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
)

app = Flask(__name__)

# Channel Access Token
line_bot_api = LineBotApi('ZtmdrSOL3jYcjtNFpoAWNnmFbRivHNUR8U+w3ftRnNF0ODl5G5w+aT3X1LEm6616qnHvJlxQBr3N9YhXRp1XyqSlWWOQRb1W9lx0UPX2FMPbVNNCIhIBwsXKRt0lvz+1hOa+Qe+IhMX7hfcenFWIJwdB04t89/1O/w1cDnyilFU=')
# Channel Secret
handler = WebhookHandler('62eb9886562f78d9402b8ba72a4f57f4')

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    print(body)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text="Hello, world!")
    line_bot_api.reply_message(
        event.reply_token,
        message)

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)