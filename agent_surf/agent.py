import time
import argparse
import xml.etree.ElementTree as ET

from langchain.chat_models.anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from crawler import Crawler
from templates import BROWSER_PROMPT_TEMPLATE

prompt_template = PromptTemplate.from_template(BROWSER_PROMPT_TEMPLATE)
llm = ChatAnthropic(model_name="claude-2", temperature=0)
# llm = ChatOpenAI(temperature=0)

chain = prompt_template | llm.bind(stop=["</command>"]) | StrOutputParser()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example.")
    parser.add_argument("--objective", type=str, help="Objective")

    args = parser.parse_args()
    # objective = """
    # "I want to go through the password reset (forgot password) for my EE account with email 'johnsmith@gmail.com'"
    # """
    start_page = "www.duckduckgo.com"

    crawler = Crawler()
    crawler.go_to_page(start_page)

    previous_command = ""
    while True:
        time.sleep(2)
        browser_content = crawler.crawl()
        res = (
            chain.invoke(
                {
                    "browser_content": "\n".join(browser_content),
                    "objective": args.objective,
                    "url": crawler.page.url,
                    "previous_command": previous_command,
                }
            )
            + "</command>"
        )
        print(res)
        root = ET.fromstring("<root>" + res + "</root>")
        thought_text = root.find("thought").text.strip()
        commands = (
            root.find("command").text.strip().split("\n")
        )  # in case multiple commands issued
        previous_command = "\n".join(commands)
        for command in commands:
            action_details = command.split()
            action = action_details[0]
            element_id = action_details[1]
            if action == "TYPESUBMIT":
                text_input = " ".join(action_details[2:]).replace('"', "")
                crawler.type(id=element_id, text=text_input)
                crawler.enter()
            elif action == "CLICK":
                crawler.click(id=element_id)
            elif action == "TYPE":
                text_input = " ".join(action_details[2:]).replace('"', "")
                crawler.type(id=element_id, text=text_input)
            elif action == "SCROLL":
                crawler.scroll(direction=action_details[1].lower())
            else:
                break
