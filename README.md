# Free ChatWithPdf Bot built with Zephyr
This Chatbot is built with Zephyr 7B Beta as its LLM and for embeddings , it uses HuggingFaceBgeEmbeddings which is again open source.
I had to download a specific version of Zephyr 7B model and could not upload it due to its huge size but it is available for free under Zephyr in HuggingFace website and its name is "zephyr-7b-beta.Q5_K_S.gguf".

There are two versions of the app available: one which uses the locally available pdf (by default its kept-Bangalore_trip) and uses Gradio for UI implementation (files used app.py and ingest.py). 
The second version helps the user by allowing them to upload the file and then ask questions based upon it.It uses files app2.py and ingest2.py and uses streamlit for UI implementation.

Overall it takes 5-6 minutes to run on a CPU .

# Attaching some screenshot below :

![WhatsApp Image 2023-10-29 at 14 55 18](https://github.com/career-tokens/ChatWithPdfOpenSourceLLM/assets/134730030/945ff593-5c4d-49f9-8275-2619f39f02a2)

![WhatsApp Image 2023-10-29 at 15 09 27](https://github.com/career-tokens/ChatWithPdfOpenSourceLLM/assets/134730030/b765aecd-ffcc-400a-b2b6-1fe76ebe32bb)

![WhatsApp Image 2023-10-29 at 15 09 44](https://github.com/career-tokens/ChatWithPdfOpenSourceLLM/assets/134730030/da3cc5d7-c8be-46a0-9e1c-659305c55cad)

