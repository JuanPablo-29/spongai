from ddgs import DDGS
import ollama
class LLM:
    def __init__(self, model, prompt_file, remember: int):
        self.memory = []
        self.model = model
        self.prompt = self.prompt_file(prompt_file)
        self.remember = remember
    
    def ask(self, content):
        messages = []
        if self.prompt.strip():
            messages.append({"role": "system", "content": self.prompt})
        messages.extend(self.memory)
        messages.append({"role": "user", "content": content})

        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        answer = response["message"]["content"]
        self.add_memory(content, answer)
        print(answer)
        return answer

    def add_memory(self, user, ai):
        if len(self.memory) > self.remember:
            self.memory.pop(0)
            self.memory.pop(0)
        self.memory.append({"role": "user", "content": user})
        self.memory.append({"role": "assistant", "content": ai})

    def prompt_file(self, file):
        with open(f"prompts/{file}.txt", "r", encoding="utf-8") as file:
            return file.read()

    def search_ask(self, content, search):
        search_results = self.search(search)
        prompt = f"""{self.prompt_file("search")}
        Search Results:
        {search_results}

        Conversation Context:
        {self.memory}

        User Question:
        {content}

        Answer:"""
        response = ollama.generate(
            model=self.model,
            prompt=prompt
        )
        answer = response["response"]
        self.add_memory(content, answer)
        return answer

    def search(self, query, num_results=10):
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            response = "\n".join([r["title"] + ": " + r["body"] for r in results])
        return response

    def router_ask(self, content):
        prompt = f"""{self.prompt_file("decide")}

        Conversation History:
        {self.memory}

        User:
        {content}
        JSON:
        """
        response = ollama.generate(
            model="mistral",
            prompt=prompt,
            options={
                "temperature": 0,
                "num_predict": 20
            }
        )
        return response["response"].strip()
