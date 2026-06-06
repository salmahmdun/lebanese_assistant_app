

import sys
import os
import gradio as gr


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.load_model import load_lebanese_model
from inference.generate import generate_response

print("⏳ Initializing Local Lebanese Assistant Chatbot UI...")
model, tokenizer = load_lebanese_model()
print("✅ Local Model Loaded Successfully!")

def respond(message, chat_history):
    if not message.strip():
        return "", chat_history
        
    chat_history.append({"role": "user", "content": str(message)})
    

    tuples_history = []
    for i in range(0, len(chat_history) - 1, 2):
        if i+1 < len(chat_history):
            tuples_history.append((chat_history[i]["content"], chat_history[i+1]["content"]))
            
    response_text = generate_response(model, tokenizer, message, tuples_history)
    
    chat_history.append({"role": "assistant", "content": response_text})
    return "", chat_history

def load_example(example_text):
    return example_text


custom_css = """
.gradio-container { background-color: #030712 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.neon-title { color: #ffffff !important; text-shadow: 0 0 10px #00f2fe, 0 0 20px #00f2fe !important; }
#lebanese_chatbot { border: 2px solid #00f2fe !important; border-radius: 16px !important; background-color: #0b0f19 !important; box-shadow: 0 0 15px rgba(0, 242, 254, 0.3) !important; }
#msg_input { background-color: #111827 !important; color: white !important; border: 1px solid #4f46e5 !important; box-shadow: 0 0 8px rgba(79, 70, 229, 0.2) !important; }
#msg_input:focus { border-color: #00f2fe !important; box-shadow: 0 0 12px rgba(0, 242, 254, 0.6) !important; }
#submit_btn { background: linear-gradient(135deg, #00f2fe, #4f46e5) !important; color: white !important; font-weight: bold !important; border: none !important; box-shadow: 0 0 12px rgba(0, 242, 254, 0.5) !important; cursor: pointer; }
.neon-example-btn { background-color: #0b0f19 !important; color: #cbd5e1 !important; border: 1px solid #bc13fe !important; border-radius: 8px !important; box-shadow: 0 0 8px rgba(188, 19, 254, 0.2) !important; transition: all 0.3s ease !important; cursor: pointer; font-size: 14px !important; padding: 10px !important; text-align: left !important; }
.neon-example-btn:hover { border-color: #00f2fe !important; box-shadow: 0 0 12px rgba(0, 242, 254, 0.6) !important; color: white !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=custom_css) as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 25px; margin-top: 15px;">
            <h1 class="neon-title" style="font-size: 34px; font-weight: 800; margin-bottom: 5px;">🇱🇧 المساعد اللبناني المطور - Lebanese Assistant 🇱🇧</h1>
            <p style="color: #9ca3af; font-size: 15px;">واجهة محادثة نيون مستقلة وموزونة تعمل محلياً على جهازك 🚀</p>
        </div>
        """
    )
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Chatbot Interface", elem_id="lebanese_chatbot", height=450, type="messages")
        
    with gr.Row():
        with gr.Column(scale=9):
            msg = gr.Textbox(show_label=False, placeholder="Type a message here or click an example below...", elem_id="msg_input", container=False)
        with gr.Column(scale=1, min_width=80):
            submit = gr.Button("إرسال", elem_id="submit_btn")
            
    gr.Markdown("<p style='color: #9ca3af; margin-top: 15px; font-weight: bold;'>💡 جرب إحدى الأمثلة التالية بلهجتك:</p>")
    examples_list = [
        "Shu y3ne API w kf byeshteghlo?",
        "Lesh l sama zar2a? fassirli bl lebnene.",
        "اعطيني نصيحة سريعة لتعلم البرمجة بالبيت.",
        "اكتبلي رسالة تهنئة مضحكة لعيد ميلاد رفيقي."
    ]
    
    with gr.Row():
        for ex_text in examples_list:
            ex_btn = gr.Button(ex_text, elem_classes="neon-example-btn")
            ex_btn.click(load_example, inputs=[ex_btn], outputs=[msg])
            
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
