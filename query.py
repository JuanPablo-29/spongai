from llm import *
import json
class Query():
    def __init__(self):
        self.llm = LLM("llama3.1:8b", "respond", 10)

    def main(self, q):
        self.query = q
        if not self.llm.mean:
            return self.parse(self.llm.router_ask(self.query, "decide"))
        else:
            return self.parse(self.llm.router_ask(self.query, "mean_decide"))
    def parse(self, result):
        print(f"Result: {result}")
        result = result.split("}")[0] + "}"
        data = json.loads(result)
        action = data["action"]
        if action == "search":
            search = data["query"]
            print("Searching...\n")
            return self.llm.search_ask(self.query, search)
        elif action == "respond":
            return self.llm.ask(self.query)
        elif action == "mean":
            self.llm.toggle_mean()
            return self.llm.ask(self.query)
        elif action == "nice":
            self.llm.toggle_mean()
            return self.llm.ask(self.query)

if __name__ == "__main__":       
    Query()