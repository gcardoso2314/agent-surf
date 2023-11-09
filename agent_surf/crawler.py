from typing import Union
from playwright.sync_api import Browser, CDPSession, Page, sync_playwright

LINK_TEMPLATE = "<link id={id} href={href}>{text}</link>"
BUTTON_TEMPLATE = "<button id={id}>{text}</button>"
TEXT_INPUT_TEMPLATE = '<input id={id} type="text" placeholder="{placeholder}" name="{name}" value="{value}">{text}</input>'
PASSWORD_INPUT_TEMPLATE = '<input id={id} type="password" placeholder="{placeholder}" name="{name}" value="{value}">{text}</input>'

TEMPLATE_MAP = {
    "link": LINK_TEMPLATE,
    "button": BUTTON_TEMPLATE,
    "text_input": TEXT_INPUT_TEMPLATE,
    "password_input": PASSWORD_INPUT_TEMPLATE,
}


def get_text_from_element_or_parent(el):
    text = el.text_content().strip()

    # If the element doesn't have text, try to get the text from its immediate parent.
    if not text:
        parent_text = el.evaluate(
            "element => element.parentElement && element.parentElement.textContent"
        )
        if parent_text:
            text = parent_text.strip()

    return text


class Crawler:
    def __init__(self):
        self.browser = sync_playwright().start().chromium.launch(headless=False)
        self.page = self.browser.new_page()
        self.page.set_viewport_size({"width": 1280, "height": 1080})
        self.page_element_buffer = {}
        self.client: CDPSession

    def go_to_page(self, url: str) -> None:
        self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    def scroll(self, direction: str) -> None:
        if direction == "up":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )

    def click(self, id: Union[str, int]) -> None:
        # Inject javascript into the page which removes the target= attribute from all links
        js = """
		links = document.getElementsByTagName("a");
		for (var i = 0; i < links.length; i++) {
			links[i].removeAttribute("target");
		}
		"""
        self.page.evaluate(js)

        element = self.page_element_buffer.get(int(id))
        if element:
            element.click()
        else:
            print("Could not find element")

    def type(self, id: Union[str, int], text: str) -> None:
        element = self.page_element_buffer.get(int(id))
        if element:
            element.type(text)
        else:
            print("Could not find element")

    def enter(self) -> None:
        self.page.keyboard.press("Enter")

    def crawl(self):
        links = self.page.query_selector_all("a")
        links_data = [
            {
                "type": "link",
                "href": el.get_attribute("href"),
                "text": el.text_content(),
                "element": el,
            }
            for el in links
        ]

        # Extract buttons
        buttons = self.page.query_selector_all("button")
        buttons_data = [
            {"type": "button", "text": el.text_content(), "element": el}
            for el in buttons
        ]

        # Extract text input boxes
        text_inputs = self.page.query_selector_all('input[type="text"]')
        text_inputs_data = [
            {
                "type": "text_input",
                "text": get_text_from_element_or_parent(el),
                "name": el.get_attribute("name"),
                "value": el.get_attribute("value"),
                "placeholder": el.get_attribute("placeholder"),
                "element": el,
            }
            for el in text_inputs
        ]

        # Extract password input boxes
        password_inputs = self.page.query_selector_all('input[type="password"]')
        password_inputs_data = [
            {
                "type": "password_input",
                "text": get_text_from_element_or_parent(el),
                "name": el.get_attribute("name"),
                "value": el.get_attribute("value"),
                "placeholder": el.get_attribute("placeholder"),
                "element": el,
            }
            for el in password_inputs
        ]

        all_content = []
        for i, element in enumerate(
            links_data + buttons_data + text_inputs_data + password_inputs_data
        ):
            all_content.append(TEMPLATE_MAP[element["type"]].format(id=i, **element))
            self.page_element_buffer[i] = element["element"]
        return all_content
