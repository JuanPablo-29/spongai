from llm import *
import json
class Query():
    def __init__(self):
        self.llm = LLM("llama3.1:8b", "respond", 20)

    def main(self, q):
        self.query = q
        return self.parse(self.llm.router_ask(self.query))

    def parse(self, result):
        print(f"Result: {result}")
        result = result.split("}")[0] + "}"
        data = json.loads(result)
        action = data["action"]
        if action == "search":
            search = data["query"]
            print("Searching...\n")
            return self.llm.search_ask(self.query, search, 5)
        elif action == "respond":
            return self.llm.ask(self.query)
            
if __name__ == "__main__":       
    Query()