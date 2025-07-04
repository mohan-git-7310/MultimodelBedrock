import streamlit as st
import boto3
import json
import base64
import requests
import io
from PIL import Image
import os
# import sounddevice as sd
# import soundfile as sf
import tempfile
from dotenv import load_dotenv
# import fitz  # PyMuPDF
import uuid

load_dotenv()

# Configure page
st.set_page_config(
    page_title="AWS Services Integration",
    page_icon="🚀",
    layout="wide"
)

# Initialize AWS clients
@st.cache_resource
def get_aws_clients():
    try:
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lex_client = boto3.client('lexv2-runtime', region_name='us-east-1')
        return bedrock, lambda_client, lex_client
    except Exception as e:
        st.error(f"Error initializing AWS clients: {str(e)}")
        return None, None, None

# Audio recording utility
def record_audio(duration=5, samplerate=16000):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, recording, samplerate)
    return temp_file.name

# Claude text generation
def generate_text(prompt, bedrock_client):
    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            })
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Claude multimodal image+question answering
def ask_claude_with_image(image_bytes, question, bedrock_client):
    try:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                            {"type": "text", "text": question}
                        ]
                    }
                ],
                "max_tokens": 500
            })
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        return f"Error analyzing image with Claude: {str(e)}"

# Image generation via Bedrock (Stable Diffusion)
def generate_image(prompt, bedrock_client):
    try:
        response = bedrock_client.invoke_model(
            modelId="stability.stable-diffusion-xl-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 10,
                "seed": 42,
                "steps": 50
            })
        )
        result = json.loads(response['body'].read())
        image_data = base64.b64decode(result['artifacts'][0]['base64'])
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        return f"Error generating image: {str(e)}"

# Lambda text summarization
def call_lambda_summarize(text, lambda_client):
    try:
        payload = {"body": json.dumps({"text": text})}
        response = lambda_client.invoke(
            FunctionName='summarize_lambda',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload),
        )
        response_payload = json.load(response['Payload'])
        return json.loads(response_payload['body']).get("summary", "No summary returned")
    except Exception as e:
        return f"Error calling Lambda: {str(e)}"

# API Gateway translation
def call_api_gateway_translate(text, direction):
    try:
        url = "https://4tud9deny0.execute-api.us-east-1.amazonaws.com/translationstage"
        payload = {"body": json.dumps({"text": text, "direction": direction})}
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            inner_body = json.loads(json.loads(response.text)["body"])
            return inner_body.get("translation", "Translation not found")
        else:
            return f"API Gateway error: {response.status_code}"
    except Exception as e:
        return f"Error calling API Gateway: {str(e)}"

# Lex audio processing
def process_audio_with_lex(audio_path, lex_client):
    try:
        bot_id = 'ZTEA8D6PJD'
        bot_alias_id = 'TSTALIASID'
        locale_id = 'en_US'
        session_id = 'streamlit-session'
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        response = lex_client.recognize_utterance(
            botId=bot_id,
            botAliasId=bot_alias_id,
            localeId=locale_id,
            sessionId=session_id,
            requestContentType='audio/l16; rate=16000; channels=1',
            responseContentType='audio/mpeg',
            inputStream=audio_bytes
        )
        return response.get("inputTranscript", "No text recognized"), response.get("audioStream")
    except Exception as e:
        return f"Error processing audio with Lex: {str(e)}", None

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Ask a question based on PDF content
def ask_question_about_pdf(pdf_text, question, bedrock_client):
    prompt = f"""You are given the following PDF content:

--- START OF PDF CONTENT ---
{pdf_text[:4000]}
--- END OF PDF CONTENT ---

Answer the question based on this content: \"{question}\"
"""
    return generate_text(prompt, bedrock_client)

# New: Chat with Bedrock Agent
def chat_with_agent(message):
    try:
        bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        session_id = str(uuid.uuid4())
        response = bedrock_agent_runtime.invoke_agent(
            agentId="JCIGJX9AQG",
            agentAliasId="XTCMMWOF2D",
            sessionId=session_id,
            inputText=message
        )
        response_text = ""
        for event in response["completion"]:
            if "chunk" in event:
                content = event["chunk"]["bytes"].decode("utf-8")
                response_text += content
        return response_text
    except Exception as e:
        return f"Error interacting with agent: {str(e)}"

# Main app
def main():
    st.title("🚀 AWS Services Integration App")

    bedrock_client, lambda_client, lex_client = get_aws_clients()
    if not all([bedrock_client, lambda_client, lex_client]):
        st.error("AWS client init failed.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Text Input", "🎤 Audio Input", "📄 PDF Input", "🖼️ Talk to Image", "🧠 Talk to Agent"
    ])

    with tab1:
        st.header("Text Input Processing")
        user_input = st.text_area("Enter your text:", height=150)
        direction = st.selectbox("Translation direction:", ["auto-en", "en-hi", "hi-en", "en-es", "es-en"])

        if st.button("Process Text", type="primary"):
            if user_input:
                user_input_lower = user_input.lower()
                if 'generate image' in user_input_lower:
                    prompt = user_input.replace('generate image', '').strip()
                    result = generate_image(prompt or "A beautiful landscape", bedrock_client)
                    st.image(result) if isinstance(result, Image.Image) else st.error(result)
                elif 'summarize' in user_input_lower:
                    text = user_input.replace('summarize', '').strip(':').strip()
                    result = call_lambda_summarize(text or "Please provide text", lambda_client)
                    st.success(result)
                elif 'translate' in user_input_lower:
                    text = user_input.replace('translate', '').strip(':').strip()
                    result = call_api_gateway_translate(text or "Hello world", direction)
                    st.success(result)
                else:
                    result = generate_text(user_input, bedrock_client)
                    st.write(result)
            else:
                st.warning("Enter some text first.")

    with tab2:
        st.header("Audio Input Processing")
        # duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
        # if st.button("Record and Process Audio"):
        #     with st.spinner("Recording and sending to Lex..."):
        #         audio_path = record_audio(duration)
        #         recognized_text, audio_response = process_audio_with_lex(audio_path, lex_client)
        #         if not recognized_text.startswith("Error"):
        #             st.success(f"Recognized: {recognized_text}")
        #             if audio_response:
        #                 audio_data = audio_response.read()
        #                 st.audio(audio_data, format='audio/mp3')
        #         else:
        #             st.error(recognized_text)

    with tab3:
        st.header("Ask Questions from PDF")
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
        question = st.text_input("Ask a question based on the PDF")

        if st.button("Ask PDF Question", type="primary") and uploaded_pdf and question:
            with st.spinner("Extracting and processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                if pdf_text.startswith("Error"):
                    st.error(pdf_text)
                else:
                    answer = ask_question_about_pdf(pdf_text, question, bedrock_client)
                    st.success(answer)
        elif not uploaded_pdf and st.button("Ask PDF Question"):
            st.warning("Please upload a PDF first.")

    with tab4:
        st.header("Talk to Image")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        question = st.text_input("What would you like to ask about the image?")

        if uploaded_image and question:
            if st.button("Analyze and Answer"):
                with st.spinner("Sending image to Claude..."):
                    image_bytes = uploaded_image.read()
                    response = ask_claude_with_image(image_bytes, question, bedrock_client)
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                    st.success(response)
        elif uploaded_image and not question:
            st.info("Please enter a question about the image.")
        elif question and not uploaded_image:
            st.warning("Please upload an image.")

    with tab5:
        st.header("Talk to Claude Agent")
        agent_input = st.text_input("Enter your message for the LLM Agent")

        if st.button("Send to Agent"):
            if agent_input:
                with st.spinner("Contacting Bedrock Agent..."):
                    agent_response = chat_with_agent(agent_input)
                    st.success(agent_response)
            else:
                st.warning("Please enter a message to send to the agent.")

if __name__ == "__main__":
    main()
