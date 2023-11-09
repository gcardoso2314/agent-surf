import time
import argparse
import xml.etree.ElementTree as ET

from langchain.chat_models.anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from agent_surf.crawler import Crawler
from agent_surf.prompts import BROWSER_PROMPT_TEMPLATE


def _build_llm_chain():
    prompt_template = PromptTemplate.from_template(BROWSER_PROMPT_TEMPLATE)
    llm = ChatAnthropic(model_name="claude-2", temperature=0)

    return prompt_template | llm.bind(stop=["</command>"]) | StrOutputParser()


def run_agent(objective, start_page):
    crawler = Crawler()
    crawler.go_to_page(start_page)

    chain = _build_llm_chain()

    previous_command = ""
    while True:
        time.sleep(2)
        browser_content = crawler.crawl()
        res = (
            chain.invoke(
                {
                    "browser_content": "\n".join(browser_content),
                    "objective": objective,
                    "url": crawler.page.url,
                    "previous_command": previous_command,
                }
            )
            + "</command>"
        )
        print(res)
        root = ET.fromstring("<root>" + res + "</root>")
        thought_text = root.find("thought").text.strip()
        print(f"Thought: {thought_text}")
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
