"""Gọi thử LLM 1 lần để kiểm tra `.env`. Không in API key."""

from career_advisor.llm import make_client

if __name__ == "__main__":
    client = make_client()
    print(f"Provider: {client.provider}, model: {client.model}")
    answer = client.complete("Trả lời đúng 1 từ: thủ đô của Việt Nam là gì?")
    print(f"Trả lời: {answer.strip()}")
    print(f"Số lượt gọi thật: {client.calls} (0 nghĩa là lấy từ cache)")
