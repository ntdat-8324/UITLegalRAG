from openai import OpenAI
from typing import List

class LLMAnswerGenerator:
    def __init__(self, api_key: str, model: str = "google/gemma-2-9b-it:free"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "Vietnamese Legal RAG",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong các văn bản pháp luật để trả lời câu hỏi của bạn."
            
        context_text = "\n\n".join([f"Văn bản {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam.
Dựa trên các văn bản pháp luật sau đây, hãy trả lời câu hỏi một cách chính xác và đầy đủ. Chỉ sử dụng thông tin từ các văn bản được cung cấp.

Văn bản tham khảo:
{context_text}

Câu hỏi: {question}

Trả lời:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Đã xảy ra lỗi khi tạo câu trả lời từ LLM."

    def generate_answer_stream(self, question: str, contexts: List[str]):
        if not contexts:
            yield "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong các văn bản pháp luật để trả lời câu hỏi của bạn."
            return
            
        context_text = "\n\n".join([f"Văn bản {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam.
Dựa trên các văn bản pháp luật sau đây, hãy trả lời câu hỏi một cách chính xác và đầy đủ. Chỉ sử dụng thông tin từ các văn bản được cung cấp.

Văn bản tham khảo:
{context_text}

Câu hỏi: {question}

Trả lời:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error calling LLM stream: {e}")
            yield "Đã xảy ra lỗi khi tạo câu trả lời từ LLM."
